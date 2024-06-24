from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        results, overall_sentiment, page_title = analyze_comments(url)
        return render_template('results.html', results=results, overall_sentiment=overall_sentiment, page_title=page_title)
    return render_template('index.html')

def analyze_comments(url):
    # Setup Chrome WebDriver
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    # Open the Reddit page
    driver.get(url)

    # Extract the title of the page
    page_title = driver.title

    # Wait for the page to load and perform initial scrolling
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    comment_count = 0
    load_more_available = True
    while load_more_available and comment_count < 1000:
        try:
            load_more_button = driver.find_element(By.XPATH, "//button[contains(., 'Ver mais comentários')]")
            load_more_button.click()
            time.sleep(3)
        except Exception as e:
            print("No more 'Load more comments' button found or error clicking it:", e)
            load_more_available = False

        comments = driver.find_elements(By.CSS_SELECTOR, "div.md")
        comment_count = len(comments)

    print(f"Total comments collected: {len(comments)}")

    # Load the saved model and the vectorizer
    model = load_model('./saved_models/sentiment_model.h5')
    with open('./saved_models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    results = []
    sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}

    for comment in comments:
        comment_text = comment.text
        comment_vectorized = vectorizer.transform([comment_text]).todense()
        prediction = model.predict(comment_vectorized)
        predicted_class = np.argmax(prediction, axis=1)
        sentiment = ['negative', 'neutral', 'positive'][predicted_class[0]]
        sentiment_counts[sentiment] += 1
        results.append({'comment': comment_text, 'sentiment': sentiment})

    driver.quit()

    # Determine the overall sentiment
    if sum(sentiment_counts.values()) == 0:
        overall_sentiment = 'No comments to analyze'
    else:
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return results, overall_sentiment, page_title

if __name__ == '__main__':
    app.run(debug=True)
