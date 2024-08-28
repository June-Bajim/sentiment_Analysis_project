from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import jsonify
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK VADER lexicon
nltk.download('vader_lexicon')

# Load or Train Model
model = None
vectorizer = None

def load_or_train_model():
    global model, vectorizer

    # Check if model and vectorizer already exist
    if os.path.exists('logistic_regression_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Model and vectorizer loaded successfully.")
    else:
        # Load dataset (replace 'your_data.csv' with your actual dataset)
        data = pd.read_csv('amazon_reviews.csv')

        # Drop specific columns (assuming we want to drop 'reviewerID' and 'asin' columns)
        columns_to_drop = ['reviewId', 'userName','thumbsUpCount','reviewCreatedVersion','at','appVersion',]
        data = data.drop(columns=columns_to_drop)

# Display the dataset after dropping columns
        print("\nData after dropping columns:")
        print(data.head())
        conditions = [
             (data['score'] >= 4),  # Example condition for positive sentiment
             (data['score'] <= 2)   # Example condition for negative sentiment
        ]

# Make sure your values are all integers or all strings
        values = [1, 0]  # Example: 1 for positive, 0 for negative

# Ensure the default value is of the same type
        default_value = -1  # Example: -1 for neutral or undefined sentiment

# Create the sentiment score column
        data['sentiment_score'] = np.select(conditions, values, default=default_value)

# Display the updated dataset
        print("\nData with sentiment scores:")
        print(data)

# Check for NaN values
        print(data['content'].isnull().sum())

# Option 1: Drop rows with NaN values
        data = data.dropna(subset=['content'])

        # Data Preprocessing
        X = data['content']
        y = data['sentiment_score']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Download NLTK resources if not already done
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        def text_preprocessing(text):
            # Function to remove emojis
            def remove_emoji(text):
                emoji_pattern = re.compile(
                    "["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F700-\U0001F77F"  # alchemical symbols
                    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    u"\U00002702-\U000027B0"  # Dingbats
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                return emoji_pattern.sub(r'', text)

            # Convert text to lowercase
        text = text.lower()

            # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

            # Remove digits
        text = re.sub(r'\d+', ' ', text)

            # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove extra whitespaces
        text = text.strip()

            # Remove emojis
        text = remove_emoji(text)

            # Tokenize text
        tokens = word_tokenize(text)

            # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

            # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        X_train_preprocessed = [' '.join(text_preprocessing(text)) for text in X_train]
        X_test_preprocessed = [' '.join(text_preprocessing(text)) for text in X_test]

        # Feature Extraction
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train_preprocessed)
        X_test_vectorized = vectorizer.transform(X_test_preprocessed)

        # Train the Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train_vectorized, y_train)
        
        y_pred = model.predict(X_test_vectorized)
        # Evaluate the model
        #X_test_vect = vectorizer.transform(X_test)
        #y_pred = model.predict(X_test_vect)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save the model and vectorizer
        with open('logistic_regression_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        print("Model and vectorizer saved successfully.")

# Load or train the model when the app starts
load_or_train_model()

# MySQL configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost:3306/flask_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    elif request.method == 'POST':
        name = request.form.get('name')
        gender = request.form.get('gender')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        new_user = User(name=name, gender=gender, email=email, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Sign up successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Email already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Email and password are required!', 'error')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Set session
            flash('Login successful!', 'success')
            return redirect(url_for('about'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/login')
def login_form():
    return render_template('login.html')


@app.route('/predictions', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        review_text = data.get('reviewText', '').strip()

        if not review_text:
            return jsonify({'error': 'No review text provided'})

        # Process the review text
        transformed_review = vectorizer.transform([review_text])
        prediction_prob = model.predict_proba(transformed_review)[0]
        prediction='negative' if prediction_prob[1]>0.5 else 'positive'
        recommendation = 'Recommended; buy the product!!' if prediction == 'positive' else 'Not Recommended'

         # Log the review and prediction
        logging.info(f"Review: {review_text}, Prediction Probabilities: {prediction_prob}, Final Prediction: {prediction}")

        response = {
            'review': review_text,
            'sentiment': prediction,
            'recommendation': recommendation
        }

        # Store response in session
        session['response'] = response
        return jsonify(response)

    # For GET requests, retrieve response from session
    response = session.get('response', None)
    return render_template('predictions.html', response=response)


@app.route('/logout')

def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)