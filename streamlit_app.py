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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Custom CSS ---
st.markdown(
    """
    <style>
    body {
        background-color: #000000; /* Black background */
        color: #FF0000; /* Red text */
        font-family: 'Times New Roman', Times, serif;
    }
    .main .block-container {
        background-color: #1a1a1a;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.5);
    }
    .stButton button {
        background-color: #FF0000;
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(255, 0, 0, 0.4);
    }
    .stButton button:hover {
        background-color: #e60000;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {
        background-color: #333333;
        color: #FFFFFF;
        border: 1px solid #FF0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header for the Streamlit app
st.title("Sentiment Analysis Dashboard")

# Select the input method
option = st.selectbox("Select Input Method", ("Link", "Dataset", "Text"))

# --- Functions for different input methods ---
# 1. Function to scrape Amazon reviews
def get_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def scrape_reviews(url, pages):
    reviews = []
    for page_no in range(1, pages + 1):
        response = requests.get(url, headers=get_headers())
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
            st.write(f"Page {page_no} failed: {response.status_code}")
            break
    return reviews

# 2. Function to preprocess text
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)

# 3. Function for sentiment analysis on text
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# 4. Function to train multiple models and display accuracy
def train_multiple_models(data):
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

    accuracies = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((model_name, accuracy))

    # Create a dataframe for accuracies
    df_accuracies = pd.DataFrame(accuracies, columns=['Model', 'Accuracy'])

    # Visualization
    st.write("### Model Accuracies")
    st.write(df_accuracies)

    # Plot the accuracy results
    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=df_accuracies, ax=ax, palette="Reds_r")
    plt.title('Model Accuracy Comparison')
    st.pyplot(fig)

    return models

# --- Process based on input option ---
if option == "Link":
    url = st.text_input("Enter Amazon Review URL")
    pages = st.number_input("Number of Pages to Scrape", min_value=1, max_value=10, value=1)

    if st.button("Scrape Reviews"):
        if url:
            reviews = scrape_reviews(url, pages)
            df_reviews = pd.DataFrame(reviews)
            st.write(df_reviews)
            df_reviews['Processed_Description'] = df_reviews['Description'].apply(preprocess_text)
            df_reviews['Sentiment'] = df_reviews['Processed_Description'].apply(analyze_sentiment)

            # Visualize sentiment distribution
            st.write("### Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment', data=df_reviews, palette='Reds_r')
            plt.title('Sentiment Distribution for Scraped Reviews')
            st.pyplot(fig)

            st.write(df_reviews[['Name', 'Rating', 'Sentiment', 'Description']])
        else:
            st.write("Please provide a valid URL.")

elif option == "Dataset":
    uploaded_file = st.file_uploader("Choose a dataset", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        data['Processed_Description'] = data['Description'].apply(preprocess_text)
        data['Sentiment'] = data['Processed_Description'].apply(analyze_sentiment)

        # Visualize sentiment distribution
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=data, palette='Reds_r')
        plt.title('Sentiment Distribution for Uploaded Dataset')
        st.pyplot(fig)

        st.write(data[['Name', 'Rating', 'Sentiment', 'Description']])

        # Train and display accuracies for multiple models
        train_multiple_models(data)

elif option == "Text":
    user_input = st.text_area("Enter text for sentiment analysis")

    if st.button("Analyze Text"):
        processed_text = preprocess_text(user_input)
        sentiment = analyze_sentiment(processed_text)
        st.write(f"Detected Sentiment: {sentiment}")
