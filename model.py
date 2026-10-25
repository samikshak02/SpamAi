import re
import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]

df = df.dropna()
df = df.drop_duplicates()

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

df["clean_message"] = df["message"].apply(
    preprocess_text
)

X = df["clean_message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Compare models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

results = []

best_model = None
best_score = -1
best_name = None

for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\nModel:", name)
    print(classification_report(
        y_test,
        predictions,
        target_names=["Safe", "Spam"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Select model using spam F1 score
    if f1 > best_score:
        best_score = f1
        best_model = pipeline
        best_name = name

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Save complete pipeline
joblib.dump(best_model, "spam_pipeline.pkl", compress=3)

print("\nBest model:", best_name)
print("Spam F1 Score:", round(best_score * 100, 2))
print("Model saved successfully!")