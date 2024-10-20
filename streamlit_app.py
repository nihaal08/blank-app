import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import nltk
from wordcloud import WordCloud

# Download NLTK components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Streamlit UI setup
st.title("SENTIMENT ANALYSIS DASHBOARD")
st.markdown("ANALYZE THE REVIEWS TO GAIN INSIGHTS ABOUT THE PRODUCT!", unsafe_allow_html=True)

# Input method options
INPUT_METHOD_OPTIONS = (
    "Link",
    "Dataset",
    "Text",
    "Retrieve Old Reviews",
    "Show All Saved Reviews",
    "Clear Database"
)
option = st.selectbox("Select Input Method", INPUT_METHOD_OPTIONS)

# Database initialization
def initialize_database():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY,
            name TEXT,
            rating TEXT,
            description TEXT,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()

initialize_database()

# Database operations
def insert_review(name, rating, description, sentiment):
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reviews (name, rating, description, sentiment) 
        VALUES (?, ?, ?, ?)
    ''', (name, rating, description, sentiment))
    conn.commit()
    conn.close()

def fetch_all_reviews():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews')
    data = cursor.fetchall()
    conn.close()
    return data

def fetch_reviews_by_name(name):
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews WHERE name LIKE ?', (f'%{name}%',))
    data = cursor.fetchall()
    conn.close()
    return data

def clear_database():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM reviews')
    conn.commit()
    conn.close()

# Web scraping with URL validation
def get_request_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost|'  # domain...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[a-fA-F0-9]*:[a-fA-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$'
    )
    return re.match(regex, url) is not None

def scrape_reviews(url, pages):
    reviews = []
    for page_number in range(1, pages + 1):
        try:
            response = requests.get(url, headers=get_request_headers())
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                boxes = soup.select('div[data-hook="review"]')
                for box in boxes:
                    reviews.append({
                        'Name': box.select_one('[class="a-profile-name"]').text if box.select_one('[class="a-profile-name"]') else 'N/A',
                        'Rating': box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A',
                        'Title': box.select_one('[data-hook="review-title"]').text if box.select_one('[data-hook="review-title"]') else 'N/A',
                        'Description': box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A'
                    })
            else:
                st.write(f"**Error:** Page {page_number} failed: {response.status_code}")
                break
        except Exception as e:
            st.write(f"**Error:** {e}")
    return reviews

# Text processing
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text)  # Remove redundant spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def train_models(data):
    X = data['Processed_Description']
    y = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': MultinomialNB()
    }

    selected_model_name = st.selectbox("Select a model to train:", list(models.keys()))
    selected_model = models[selected_model_name]
    
    # Hyperparameter tuning example
    if selected_model_name == 'Logistic Regression':
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
    elif selected_model_name == 'Random Forest':
        param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
    elif selected_model_name == 'SVM':
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif selected_model_name == 'KNN':
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif selected_model_name == 'Naive Bayes':
        param_grid = {}

    grid_search = GridSearchCV(selected_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    df_metrics = pd.DataFrame({
        'Model': [selected_model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })

    # Toggle visibility for model performance comparison
    if st.checkbox("Show Model Performance Comparison"):
        st.write("### MODEL PERFORMANCE")
        st.write(df_metrics)

        fig, ax = plt.subplots()
        sns.barplot(x='Accuracy', y='Model', data=df_metrics, ax=ax, palette="Reds_r")
        plt.title('Model Accuracy Comparison')
        st.pyplot(fig)

    # Word Clouds for Sentiment Analysis
    sentiment_groups = data.groupby('Sentiment')['Processed_Description'].apply(lambda x: ' '.join(x)).reset_index()

    st.write("### Word Clouds for Sentiment Analysis")
    cols = st.columns(len(sentiment_groups))
    for i, sentiment in enumerate(sentiment_groups['Sentiment']):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_groups[sentiment_groups['Sentiment'] == sentiment]['Processed_Description'].values[0])
        
        with cols[i]:
            st.subheader(f'Word Cloud for {sentiment} Reviews')
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

    return best_model

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']

    insights = []
    
    # Enhanced insights
    insights.append(f"Total Reviews Analyzed: {len(data)}")
    insights.append(f"Positive Reviews: {len(positive_reviews)} ({(len(positive_reviews) / len(data) * 100):.2f}%)")
    insights.append(f"Negative Reviews: {len(negative_reviews)} ({(len(negative_reviews) / len(data) * 100):.2f}%)")
    insights.append(f"Neutral Reviews: {len(data) - len(positive_reviews) - len(negative_reviews)}")

    return insights

# Processing User Inputs
if option == "Link":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    url_input = st.text_input("Enter Amazon Review URL:")
    pages_input = st.number_input("Pages to Scrape:", 1, 50, 1)

    if st.button("SCRAPE REVIEWS"):
        if is_valid_url(url_input):
            scraped_reviews = scrape_reviews(url_input, pages_input)
            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### SCRAPED REVIEWS")
            st.write(df_reviews)

            df_reviews['Processed_Description'] = df_reviews['Description'].apply(preprocess_text)
            df_reviews['Sentiment'] = df_reviews['Processed_Description'].apply(analyze_sentiment)

            for _, row in df_reviews.iterrows():
                insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])

            st.write("### SENTIMENT DISTRIBUTION")
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment', data=df_reviews, palette='Reds_r')
            plt.title('Sentiment Count')
            st.pyplot(fig)

            insights = generate_insights(df_reviews)
            st.write("### INSIGHTS")
            for insight in insights:
                st.write(insight)

            st.write("### DETAILED DATA")
            st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Description']])
        else:
            st.write("**Please provide a valid Amazon URL.**")

elif option == "Dataset":
    st.header("UPLOAD DATASET")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### UPLOADED DATA")
        st.write(data)

        data['Processed_Description'] = data['Description'].apply(preprocess_text)
        data['Sentiment'] = data['Processed_Description'].apply(analyze_sentiment)

        for _, row in data.iterrows():
            insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])

        st.write("### SENTIMENT DISTRIBUTION")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=data, palette='Reds_r')
        plt.title('Sentiment Count')
        st.pyplot(fig)

        insights = generate_insights(data)
        st.write("### INSIGHTS")
        for insight in insights:
            st.write(insight)

        st.write("### TRAIN MODELS")
        train_models(data)

elif option == "Text":
    st.header("ANALYZE CUSTOM TEXT")
    user_input_text = st.text_area("Enter text:")

    if st.button("ANALYZE TEXT"):
        processed_text = preprocess_text(user_input_text)
        sentiment_result = analyze_sentiment(processed_text)
        blob = TextBlob(processed_text)
        polarity_score = blob.sentiment.polarity
        
        st.write(f"**Sentiment:** {sentiment_result}")
        st.write(f"**Polarity Score:** {polarity_score:.2f}")
        
elif option == "Retrieve Old Reviews":
    st.header("SEARCH OLD REVIEWS")
    search_name = st.text_input("Enter Name to Search:")

    if st.button("FETCH REVIEWS"):
        if search_name:
            reviews = fetch_reviews_by_name(search_name)
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
                st.write("### RETRIEVED REVIEWS")
                st.write(df_reviews)
            else:
                st.write("**No reviews found.**")
        else:
            st.write("**Please enter a name.**")

elif option == "Clear Database":
    st.header("CLEAR ALL REVIEWS")
    if st.button("CONFIRM CLEAR"):
        clear_database()
        st.success("**All reviews have been cleared.**")

elif option == "Show All Saved Reviews":
    st.header("VIEW ALL SAVED REVIEWS")
    if st.button("SHOW REVIEWS"):
        reviews = fetch_all_reviews()
        if reviews:
            df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
            st.write("### ALL REVIEWS")
            st.write(df_reviews)
        else:
            st.write("**No reviews found.**")
