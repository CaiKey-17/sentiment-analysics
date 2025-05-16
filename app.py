import re
import nltk
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')


# Load CSV into DataFrame
df = pd.read_csv("sentimentdataset.csv")



def setup_sentiment_sets():
    """Convert sentiment lists into sets for optimized membership testing."""
    positive_set = {
        'positive', 'happiness', 'joy', 'love', 'amusement', 'enjoyment', 'admiration',
        'affection', 'awe', 'acceptance', 'adoration', 'anticipation', 'calmness',
        'excitement', 'kind', 'pride', 'elation', 'euphoria', 'contentment', 'serenity',
        'gratitude', 'hope', 'hopeful', 'empowerment', 'compassion', 'compassionate',
        'tenderness', 'arousal', 'enthusiasm', 'fulfillment', 'reverence', 'curiosity',
        'determination', 'zest', 'proud', 'grateful', 'empathetic', 'playful',
        'free-spirited', 'inspired', 'confident', 'thrill', 'overjoyed', 'inspiration',
        'motivation', 'joyfulreunion', 'satisfaction', 'blessed', 'reflection',
        'appreciation', 'confidence', 'accomplishment', 'wonderment', 'optimism',
        'enchantment', 'intrigue', 'playfuljoy', 'mindfulness', 'dreamchaser', 'elegance',
        'whimsy', 'harmony', 'creativity', 'radiance', 'wonder', 'rejuvenation', 'coziness',
        'adventure', 'melodic', 'festivejoy', 'innerjourney', 'freedom', 'dazzle',
        'adrenaline', 'artisticburst', 'culinaryodyssey', 'resilience', 'immersion',
        'spark', 'marvel', 'positivity', 'kindness', 'friendship', 'success', 'exploration',
        'amazement', 'romance', 'captivation', 'tranquility', 'grandeur', 'emotion',
        'energy', 'celebration', 'charm', 'ecstasy', 'colorful', 'hypnotic', 'connection',
        'iconic', 'journey', 'engagement', 'touched', 'triumph', 'heartwarming',
        'solace', 'breakthrough', 'joy in baking', 'envisioning history', 'imagination',
        'vibrancy', 'mesmerizing', 'culinary adventure', 'winter magic', 'thrilling journey',
        "nature's beauty", 'celestial wonder', 'creative inspiration', 'runway creativity',
        "ocean's freedom", 'relief', 'happy'
    }
    negative_set = {
        'negative', 'anger', 'fear', 'sadness', 'disgust', 'disappointed', 'bitter',
        'shame', 'despair', 'grief', 'loneliness', 'jealousy', 'resentment', 'frustration',
        'boredom', 'anxiety', 'intimidation', 'helplessness', 'envy', 'regret',
        'indifference', 'numbness', 'melancholy', 'bitterness', 'yearning', 'fearful',
        'apprehensive', 'overwhelmed', 'jealous', 'devastated', 'frustrated', 'envious',
        'dismissive', 'bittersweet', 'heartbreak', 'betrayal', 'suffering', 'emotionalstorm',
        'isolation', 'disappointment', 'lostlove', 'exhaustion', 'sorrow', 'darkness',
        'desperation', 'ruins', 'desolation', 'loss', 'heartache', 'solitude', 'obstacle',
        'sympathy', 'pressure', 'miscalculation', 'challenge', 'embarrassed', 'sad',
        'hate', 'bad'
    }
    return positive_set, negative_set

def categorize_sentiment(sentiment, positive_set, negative_set):
    """Categorize a given sentiment as Positive, Negative, or Neutral."""
    sentiment = sentiment.strip().lower()
    if sentiment in positive_set:
        return 'Positive'
    elif sentiment in negative_set:
        return 'Negative'
    else:
        return 'Neutral'


positive_set, negative_set = setup_sentiment_sets()
df['Sentiment_Group'] = df['Sentiment'].apply(lambda s: categorize_sentiment(s, positive_set, negative_set))
df.dropna(subset=['Sentiment_Group'], inplace=True)
df['Platform'] = df['Platform'].str.strip()





## Correlation Heatmap
corr = df[['Retweets', 'Likes', 'Day', 'Hour']].corr()


## Sentiment by Platform
platform_sentiment = df.groupby(['Platform', 'Sentiment_Group']).size().reset_index(name='Count')




## Correlation Heatmap
corr = df[['Retweets', 'Likes', 'Day', 'Hour']].corr()



# Map sentiment groups to numeric labels
sentiment_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
df['label'] = df['Sentiment_Group'].map(sentiment_map)
print(df.columns)

def clean_text(text, stop_words, lemmatizer, url_pattern, non_alpha_pattern):
    """Clean input text by lowering case, removing URLs, non-alphanumeric characters, stopwords and applying lemmatization."""
    text = text.lower()
    text = url_pattern.sub(" ", text)
    text = non_alpha_pattern.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Compile regex patterns for performance
url_pattern = re.compile(r"http\S+|www\S+|@\S+")
non_alpha_pattern = re.compile(r"[^a-z0-9\s]")

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df['clean_text'] = df['Text'].apply(lambda x: clean_text(x, stop_words, lemmatizer, url_pattern, non_alpha_pattern))
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Transform text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
# Train SVC model
svc_model = SVC()
svc_model.fit(X_train_tfidf, y_train)

# Predict using the trained model
y_pred_svc = svc_model.predict(X_test_tfidf)


results = []

def get_metrics_dict(model_name, y_true, y_pred):
    """Return a dict of performance metrics for a given model's predictions."""
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]
    }

    return metrics

# SVC
param_dist_svc = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
}
svc = SVC(random_state=42)
random_search_svc = RandomizedSearchCV(
    svc, param_distributions=param_dist_svc, n_iter=30, cv=5,
    verbose=2, random_state=42, n_jobs=-1
)

random_search_svc.fit(X_train_tfidf, y_train)
best_svc_model = random_search_svc.best_estimator_
y_pred_svc = best_svc_model.predict(X_test_tfidf)

print("---- SVC ----")
print("Best Parameters:", random_search_svc.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


results.append(get_metrics_dict("SVC", y_test, y_pred_svc))

# Convert list of dicts to DataFrame
df_results = pd.DataFrame(results)

# Sort by accuracy (descending) if you like
df_results = df_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)



def predict_text(text, vectorizer, model, sentiment_map):
    """Predict the sentiment of a new text input."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    url_pattern = re.compile(r"http\S+|www\S+|@\S+")
    non_alpha_pattern = re.compile(r"[^a-z0-9\s]")

    cleaned = clean_text(text, stop_words, lemmatizer, url_pattern, non_alpha_pattern)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    # Reverse the mapping from numeric label to sentiment group
    reverse_map = {v: k for k, v in sentiment_map.items()}
    return reverse_map[pred]

models = {
    "result": best_svc_model,
}
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict sentiment from a single provided text."""
    # Get data from POST request
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    example_text = data['text']
    predictions = {}

    # Loop through the models and predict sentiment for the provided text
    for model_name, model in models.items():
        pred = predict_text(example_text, tfidf, model, sentiment_map)
        predictions[model_name] = pred

    # Return the predictions as a JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

