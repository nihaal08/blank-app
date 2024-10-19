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
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

st.title("SENTIMENT ANALYSIS DASHBOARD")
st.markdown("-----------Analyze product reviews to gain insights!----------")
INPUT_METHOD_OPTIONS = (
    "Link",
    "Dataset",
    "Text",
    "Retrieve Old Reviews",
    "Show All Saved Reviews",
    "Clear Database"
)
option = st.selectbox("Select Input Method", INPUT_METHOD_OPTIONS)

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

def get_request_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def scrape_reviews(url, pages):
    reviews = []
    for page_number in range(1, pages + 1):
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
    return reviews

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

    st.write("### Model Performance")
    st.write(df_metrics)

    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=df_metrics, ax=ax, palette="Reds_r")
    plt.title('Model Accuracy Comparison')
    st.pyplot(fig)

    return models

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']

    insights = []

    if len(positive_reviews) > 0:
        insights.append(f"{len(positive_reviews)} positive reviews found.")

    if len(negative_reviews) > 0:
        insights.append(f"{len(negative_reviews)} negative reviews found.")

    if not insights:
        insights.append("No insights available.")

    return insights

# Processing User Inputs
if option == "Link":
    st.header("Scrape Reviews from Amazon")
    url_input = st.text_input("Enter Amazon Review URL:")
    pages_input = st.number_input("Pages to Scrape:", 1, 50, 1) 

    if st.button("Scrape Reviews"):
        if url_input:
            scraped_reviews = scrape_reviews(url_input, pages_input)
            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### Scraped Reviews")
            st.write(df_reviews)

            df_reviews['Processed_Description'] = df_reviews['Description'].apply(preprocess_text)
            df_reviews['Sentiment'] = df_reviews['Processed_Description'].apply(analyze_sentiment)

            for _, row in df_reviews.iterrows():
                insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])
            
            st.write("### Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment', data=df_reviews, palette='Reds_r')
            plt.title('Sentiment Count')
            st.pyplot(fig)

            insights = generate_insights(df_reviews)
            st.write("### Insights")
            for insight in insights:
                st.write(insight)

            st.write("### Detailed Data")
            st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Description']])
        else:
            st.write("**Please provide a valid URL.**")

elif option == "Dataset":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(data)

        data['Processed_Description'] = data['Description'].apply(preprocess_text)
        data['Sentiment'] = data['Processed_Description'].apply(analyze_sentiment)

        for _, row in data.iterrows():
            insert_review(row['Name'], row['Rating'], row['Description'], row['Sentiment'])

        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=data, palette='Reds_r')
        plt.title('Sentiment Count')
        st.pyplot(fig)

        insights = generate_insights(data)
        st.write("### Insights")
        for insight in insights:
            st.write(insight)

        st.write("### Train Models")
        train_models(data)

elif option == "Text":
    st.header("Analyze Custom Text")
    user_input_text = st.text_area("Enter text:")

    if st.button("Analyze Text"):
        processed_text = preprocess_text(user_input_text)
        sentiment_result = analyze_sentiment(processed_text)
        blob = TextBlob(processed_text)
        polarity_score = blob.sentiment.polarity
        
        st.write(f"**Sentiment:** {sentiment_result}")
        st.write(f"**Polarity Score:** {polarity_score:.2f}")
        
elif option == "Retrieve Old Reviews":
    st.header("Search Old Reviews")
    search_name = st.text_input("Enter Name to Search:")

    if st.button("Fetch Reviews"):
        if search_name:
            reviews = fetch_reviews_by_name(search_name)
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
                st.write("### Retrieved Reviews")
                st.write(df_reviews)
            else:
                st.write("**No reviews found.**")
        else:
            st.write("**Please enter a name.**")

elif option == "Clear Database":
    st.header("Clear All Reviews")
    if st.button("Confirm Clear"):
        clear_database()
        st.success("**All reviews have been cleared.**")

elif option == "Show All Saved Reviews":
    st.header("View All Saved Reviews")
    if st.button("Show Reviews"):
        reviews = fetch_all_reviews()
        if reviews:
            df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Description", "Sentiment"])
            st.write("### All Reviews")
            st.write(df_reviews)
        else:
            st.write("**No reviews found.**")
