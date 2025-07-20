from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Print current directory (for debugging)
print("Current Working Directory:", os.getcwd())

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess dataset
raw_mail_data = pd.read_csv(r'C:\Users\aswin\Downloads\mail_data.csv')
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Convert labels: 'spam' = 0, 'ham' = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate features and labels
X = mail_data['Message']
y = mail_data['Category'].astype('int')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_features, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    input_data = [message]
    vectorized_input = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_input)[0]
    result = "SPAM" if prediction == 0 else "NOT SPAM"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
