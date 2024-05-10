# importing libraries
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load your dataset and preprocess it
dataset = pd.read_csv("hatedata.csv")
dataset["labels"] = dataset["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate or Offensive Language"})

stopwords = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://.+|www\..+', '', text)  # Use a raw string literal (r'...')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data = dataset[["tweet", "labels"]].copy()  # Use .copy() to avoid SettingWithCopyWarning
data["tweet"] = data["tweet"].apply(clean_data)

X = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Define your route and function for the webpage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        result = detect_hate_speech(text)
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

def detect_hate_speech(text):
    text = clean_data(text)
    data1 = cv.transform([text]).toarray()
    prediction = dt.predict(data1)
    if prediction == "Hate Speech":
        return "Hate speech detected!"
    elif prediction == "Offensive Language":
        return "Offensive language detected!"
    else:
        return "No hate or offensive language detected."

if __name__ == '__main__':
    app.run(debug=True)
