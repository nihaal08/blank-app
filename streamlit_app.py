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
from googletrans import Translator
from collections import Counter

# NLTK resource downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize resources
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Streamlit UI Title
st.markdown("<h1 style='text-align: center;'>SENTIMENT ANALYSIS DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>-----------ANALYZE PRODUCT REVIEWS TO GAIN INSIGHTS!----------</h2>", unsafe_allow_html=True)

# Input method selection
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
def execute_db_query(query, params=None, fetch=False):
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    conn.commit()
    result = cursor.fetchall() if fetch else None
    conn.close()
    return result

def insert_review(name, rating, description, sentiment):
    execute_db_query('''
        INSERT INTO reviews (name, rating, description, sentiment) 
        VALUES (?, ?, ?, ?)
    ''', (name, rating, description, sentiment))

def fetch_all_reviews():
    return execute_db_query('SELECT * FROM reviews', fetch=True)

def fetch_reviews_by_name(name):
    return execute_db_query('SELECT * FROM reviews WHERE name LIKE ?', (f'%{name}%',), fetch=True)

def clear_database():
    execute_db_query('DELETE FROM reviews')

# Web scraping and data processing
def get_request_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def scrape_reviews(url, pages):
    reviews = []
    for page_number in range(1, pages + 1):
        try:
            response = requests.get(url, headers=get_request_headers())
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, 'lxml')
            boxes = soup.select('div[data-hook="review"]')
            for box in boxes:
                name = (
                    box.select_one('[class="a-profile-name"]').text 
                    if box.select_one('[class="a-profile-name"]') 
                    else 'N/A'
                )
                rating = (
                    box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] 
                    if box.select_one('[data-hook="review-star-rating"]') 
                    else 'N/A'
                )
                description = (
                    box.select_one('[data-hook="review-body"]').text.strip() 
                    if box.select_one('[data-hook="review-body"]')
                    else 'N/A'
                )
                reviews.append({
                    'Name': name,
                    'Rating': rating,
                    'Description': description
                })
        except requests.RequestException as e:
            st.error(f"Error fetching page {page_number}: {e}")
            break

    return reviews

def preprocess_text(text):
    text = emoji.demojize(text)
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

def translate_text(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

def train_models(data):
    X = data['Processed_Description']
    y = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
    
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    # Avoid data leakage
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
    st.markdown("<h2 style='text-align: center;'>MODEL PERFORMANCE</h2>", unsafe_allow_html=True)
    st.write(df_metrics)

    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=df_metrics, ax=ax, palette="Reds_r")
    plt.title('Model Accuracy Comparison')
    st.pyplot(fig)

    return models

def generate_insights(data):
    total_reviews = len(data)
    sentiment_counts = data['Sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)

    # Identify commonly used words
    all_words = ' '.join(data['Processed_Description'])
    word_counts = Counter(all_words.split())
    common_words = word_counts.most_common(10)

    insights = {
        "total_reviews": total_reviews,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "common_words": common_words
    }

    return insights

# Processing User Inputs
if option == "Link":
    st.markdown("<h2 style='text-align: center;'>SCRAPE REVIEWS FROM AMAZON</h2>", unsafe_allow_html=True)
    url_input = st.text_input("Enter Amazon Review URL:")
    pages_input = st.number_input("Pages to Scrape:", 1, 50, 1)

    if st.button("Scrape Reviews"):
        if url_input:
            scraped_reviews = scrape_reviews(url_input, pages_input)
            if scraped_reviews:
                df_reviews = pd.DataFrame(scraped_reviews)
                st.markdown("<h3 style='text-align: center;'>SCRAPED REVIEWS</h3>", unsafe_allow_html=True)
                st.write(df_reviews)

                # Translate and process for sentiment analysis
                df_reviews['Translated_Description'] = df_reviews['Description'].apply(translate_text)
                df_reviews['Processed_Description'] = df_reviews['Translated_Description'].apply(preprocess_text)
                df_reviews['Sentiment'] = df_reviews['Processed_Description'].apply(analyze_sentiment)

                for _, row in df_reviews.iterrows():
                    insert_review(row['Name'], row['Rating'], row['Translated_Description'], row['Sentiment'])

                # Display insights after processing
                insights = generate_insights(df_reviews)
                st.markdown("<h3 style='text-align: center;'>INSIGHTS</h3>", unsafe_allow_html=True)
                st.write(f"**Total Reviews:** {insights['total_reviews']}")
                st.write(f"**Positive Reviews:** {insights['positive_count']}")
                st.write(f"**Negative Reviews:** {insights['negative_count']}")
                st.write(f"**Neutral Reviews:** {insights['neutral_count']}")
                st.write("**Most Common Words:**")
                st.write(insights['common_words'])

                st.markdown("<h3 style='text-align: center;'>DETAILED DATA</h3>", unsafe_allow_html=True)
                st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Translated_Description']])
            else:
                st.write("**No reviews were found.**")
        else:
            st.write("**Please provide a valid URL.**")

elif option == "Dataset":
    st.markdown("<h2 style='text-align: center;'>UPLOAD DATASET</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown("<h3 style='text-align: center;'>UPLOADED DATA</h3>", unsafe_allow_html=True)
            st.write(data)

            # Translate and process for sentiment analysis
            data['Translated_Description'] = data['Description'].apply(translate_text)
            data['Processed_Description'] = data['Translated_Description'].apply(preprocess_text)
            data['Sentiment'] = data['Processed_Description'].apply(analyze_sentiment)

            for _, row in data.iterrows():
                insert_review(row['Name'], row['Rating'], row['Translated_Description'], row['Sentiment'])

            st.markdown("<h3 style='text-align: center;'>SENTIMENT DISTRIBUTION</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment', data=data, palette='Reds_r')
            plt.title('Sentiment Count')
            st.pyplot(fig)

            st.markdown("<h3 style='text-align: center;'>DETAILED DATA</h3>", unsafe_allow_html=True)
            st.write(data[['Name', 'Rating', 'Sentiment', 'Translated_Description']])

            # Generate insights after review processing
            insights = generate_insights(data)
            st.markdown("<h3 style='text-align: center;'>INSIGHTS</h3>", unsafe_allow_html=True)
            st.write(f"**Total Reviews:** {insights['total_reviews']}")
            st.write(f"**Positive Reviews:** {insights['positive_count']}")
            st.write(f"**Negative Reviews:** {insights['negative_count']}")
            st.write(f"**Neutral Reviews:** {insights['neutral_count']}")
            st.write("**Most Common Words:**")
            st.write(insights['common_words'])

            st.markdown("<h3 style='text-align: center;'>TRAIN MODELS</h3>", unsafe_allow_html=True)
            train_models(data)

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif option == "Text":
    st.markdown("<h2 style='text-align: center;'>ANALYZE CUSTOM TEXT</h2>", unsafe_allow_html=True)
    user_input_text = st.text_area("Enter text:")

    if st.button("Analyze Text"):
        translated_text = translate_text(user_input_text)
        processed_text = preprocess_text(translated_text)
        sentiment_result = analyze_sentiment(processed_text)
        blob = TextBlob(processed_text)
        polarity_score = blob.sentiment.polarity

        st.write(f"**Translated Text:** {translated_text}")
        st.write(f"**Sentiment:** {sentiment_result}")
        st.write(f"**Polarity Score:** {polarity_score:.2f}")

elif option == "Retrieve Old Reviews":
    st.markdown("<h2 style='text-align: center;'>SEARCH OLD REVIEWS</h2>", unsafe_allow_html=True)
    search_name = st.text_input("Enter Name to Search:")

    if st.button("Fetch Reviews"):
        if search_name:
            reviews = fetch_reviews_by_name(search_name)
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
                st.markdown("<h3 style='text-align: center;'>RETRIEVED REVIEWS</h3>", unsafe_allow_html=True)
                st.write(df_reviews)
            else:
                st.write("**No reviews found.**")
        else:
            st.write("**Please enter a name.**")

elif option == "Clear Database":
    st.markdown("<h2 style='text-align: center;'>CLEAR ALL REVIEWS</h2>", unsafe_allow_html=True)
    if st.button("Confirm Clear"):
        clear_database()
        st.success("**All reviews have been cleared.**")

elif option == "Show All Saved Reviews":
    st.markdown("<h2 style='text-align: center;'>VIEW ALL SAVED REVIEWS</h2>", unsafe_allow_html=True)
    if st.button("Show Reviews"):
        reviews = fetch_all_reviews()
        if reviews:
            df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
            st.markdown("<h3 style='text-align: center;'>ALL REVIEWS</h3>", unsafe_allow_html=True)
            st.write(df_reviews)
        else:
            st.write("**No reviews found.**")
