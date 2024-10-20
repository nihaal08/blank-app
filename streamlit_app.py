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
from wordcloud import WordCloud
import sqlite3
import nltk

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
# ... (insert_review, fetch_all_reviews, fetch_reviews_by_name, clear_database)

# URL validation
def is_valid_url(url):
    return re.match(r'https:\/\/www\.amazon\.(com|co\.uk|ca|de|fr|it|es|jp)\/.*', url)

# Web scraping
# ... (get_request_headers, scrape_reviews)

# Text processing
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text)  # Remove redundant spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)

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

    # Hyperparameter tuning with GridSearchCV for Logistic Regression as an example
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    df_metrics = pd.DataFrame({'Model': ['Logistic Regression'], 
                                'Accuracy': [accuracy],
                                'Precision': [precision],
                                'Recall': [recall],
                                'F1 Score': [f1]})

    st.write("### MODEL PERFORMANCE")
    st.write(df_metrics)

    return model

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']
    
    insights = [
        f"Total Reviews Analyzed: {len(data)}",
        f"Positive Reviews: {len(positive_reviews)} ({(len(positive_reviews) / len(data) * 100):.2f}%)",
        f"Negative Reviews: {len(negative_reviews)} ({(len(negative_reviews) / len(data) * 100):.2f}%)",
        f"Neutral Reviews: {len(data) - len(positive_reviews) - len(negative_reviews)}"
    ]
    return insights

# Additional Visualization: Word Cloud
def plot_word_cloud(data):
    text = ' '.join(data['Processed_Description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Processing User Inputs
if option == "Link":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    url_input = st.text_input("Enter Amazon Review URL:")
    pages_input = st.number_input("Pages to Scrape:", 1, 50, 1)

    if st.button("SCRAPE REVIEWS"):
        try:
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

                # Plot Word Cloud
                st.write("### WORD CLOUD")
                plot_word_cloud(df_reviews)

                st.write("### DETAILED DATA")
                st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Description']])
            else:
                st.write("**Please provide a valid Amazon URL.**")
        except Exception as e:
            st.write(f"**Error:** {e}")

elif option == "Dataset":
    st.header("UPLOAD DATASET")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
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

        except Exception as e:
            st.write(f"**Error:** {e}")

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
        try:
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
        except Exception as e:
            st.write(f"**Error:** {e}")

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
