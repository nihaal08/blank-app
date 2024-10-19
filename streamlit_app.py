import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
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

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to initialize the database
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

# Function to insert review into the database
def insert_review(name, rating, description, sentiment):
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reviews (name, rating, description, sentiment) 
        VALUES (?, ?, ?, ?)
    ''', (name, rating, description, sentiment))
    conn.commit()
    conn.close()

# Function to fetch all saved reviews
def fetch_all_reviews():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews')
    data = cursor.fetchall()
    conn.close()
    return data

# Function to fetch reviews by name
def fetch_reviews_by_name(name):
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews WHERE name LIKE ?', (f'%{name}%',))
    data = cursor.fetchall()
    conn.close()
    return data

# Function to clear the database
def clear_database():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM reviews')
    conn.commit()
    conn.close()

# Function to get request headers
def get_request_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

# Function to scrape reviews from the given URL
def scrape_reviews(url, pages):
    reviews = []
    for page_number in range(1, pages + 1):
        try:
            response = requests.get(url, headers=get_request_headers())
            response.raise_for_status()  # Raise an exception for HTTP error codes
            soup = BeautifulSoup(response.text, 'lxml')
            boxes = soup.select('div[data-hook="review"]')
            for box in boxes:
                reviews.append({
                    'Name': box.select_one('[class="a-profile-name"]').text if box.select_one('[class="a-profile-name"]') else 'N/A',
                    'Rating': box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A',
                    'Title': box.select_one('[data-hook="review-title"]').text if box.select_one('[data-hook="review-title"]') else 'N/A',
                    'Description': box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A'
                })
        except requests.HTTPError as e:
            st.error(f"HTTP error occurred: {e}")
            break
        except Exception as e:
            st.error(f"An error occurred: {e}")
            break
    return reviews

# Initialize NLTK tools
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

# Function to train the sentiment analysis models
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

    metrics = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics.append((model_name, accuracy, precision, recall, f1))

    df_metrics = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    st.write("### MODEL PERFORMANCE", unsafe_allow_html=True)
    st.write(df_metrics)

    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=df_metrics, ax=ax, palette="Reds_r")
    plt.title('Model Accuracy Comparison')
    st.pyplot(fig)

# Function to generate insights from the data
def generate_insights(data):
    insights = {
        "Total Reviews": len(data),
        "Positive Reviews": len(data[data['Sentiment'] == 'Positive']),
        "Negative Reviews": len(data[data['Sentiment'] == 'Negative']),
        "Neutral Reviews": len(data[data['Sentiment'] == 'Neutral']),
    }
    return insights

# Streamlit UI configuration
st.title("SENTIMENT ANALYSIS DASHBOARD")
st.markdown("<h2 style='text-align: center;'>ANALYZE PRODUCT REVIEWS TO GAIN INSIGHTS</h2>", unsafe_allow_html=True)
INPUT_METHOD_OPTIONS = (
    "Link",
    "Dataset",
    "Text",
    "Retrieve Old Reviews",
    "Show All Saved Reviews",
    "Clear Database"
)
option = st.selectbox("SELECT INPUT METHOD", INPUT_METHOD_OPTIONS)

# Initialize the database
initialize_database()

# Main logic for different input methods
if option == "Link":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    url_input = st.text_input("ENTER AMAZON REVIEW URL:")
    pages_input = st.number_input("PAGES TO SCRAPE:", 1, 50, 1)

    if st.button("SCRAPE REVIEWS"):
        if url_input:
            scraped_reviews = scrape_reviews(url_input, pages_input)
            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### SCRAPED REVIEWS")
            st.write(df_reviews)

            df_reviews['Processed_Description'] = df_reviews['Description'].apply(preprocess_text)
            df_reviews['Sentiment'] = df_reviews['Processed_Description'].apply(analyze_sentiment)

            # Insert reviews into the database
            for _, row in df_reviews.iterrows():
                insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])

            # Displaying sentiment distribution
            st.write("### SENTIMENT DISTRIBUTION")
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment', data=df_reviews, palette='Reds_r')
            plt.title('Sentiment Count')
            st.pyplot(fig)

            # Generating insights
            insights = generate_insights(df_reviews)
            st.write("### INSIGHTS")
            for key, value in insights.items():
                st.write(f"**{key}:** {value}")

            st.write("### DETAILED DATA")
            st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Description']])
        else:
            st.error("**PLEASE PROVIDE A VALID URL.**")

elif option == "Dataset":
    st.header("UPLOAD DATASET")
    uploaded_file = st.file_uploader("CHOOSE A CSV FILE", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### UPLOADED DATA")
        st.write(data)

        data['Processed_Description'] = data['Description'].apply(preprocess_text)
        data['Sentiment'] = data['Processed_Description'].apply(analyze_sentiment)

        # Insert reviews into the database
        for _, row in data.iterrows():
            insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])

        st.write("### SENTIMENT DISTRIBUTION")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=data, palette='Reds_r')
        plt.title('Sentiment Count')
        st.pyplot(fig)

        # Generating insights
        insights = generate_insights(data)
        st.write("### INSIGHTS")
        for key, value in insights.items():
            st.write(f"**{key}:** {value}")

        # Train models
        st.write("### TRAIN MODELS")
        train_models(data)

elif option == "Text":
    st.header("ANALYZE CUSTOM TEXT")
    user_input_text = st.text_area("ENTER TEXT:")

    if st.button("ANALYZE TEXT"):
        processed_text = preprocess_text(user_input_text)
        sentiment_result = analyze_sentiment(processed_text)
        blob = TextBlob(processed_text)
        polarity_score = blob.sentiment.polarity

        st.write(f"**SENTIMENT:** {sentiment_result}")
        st.write(f"**POLARITY SCORE:** {polarity_score:.2f}")

elif option == "Retrieve Old Reviews":
    st.header("SEARCH OLD REVIEWS")
    search_name = st.text_input("ENTER NAME TO SEARCH:")

    if st.button("FETCH REVIEWS"):
        if search_name:
            reviews = fetch_reviews_by_name(search_name)
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
                st.write("### RETRIEVED REVIEWS")
                st.write(df_reviews)
            else:
                st.warning("**NO REVIEWS FOUND.**")
        else:
            st.error("**PLEASE ENTER A NAME.**")

elif option == "Clear Database":
    st.header("CLEAR ALL REVIEWS")
    if st.button("CONFIRM CLEAR"):
        clear_database()
        st.success("**ALL REVIEWS HAVE BEEN CLEARED.**")

elif option == "Show All Saved Reviews":
    st.header("VIEW ALL SAVED REVIEWS")
    if st.button("SHOW REVIEWS"):
        reviews = fetch_all_reviews()
        if reviews:
            df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
            st.write("### ALL REVIEWS")
            st.write(df_reviews)
        else:
            st.warning("**NO REVIEWS FOUND.**")
