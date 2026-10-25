import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Clean dataset
df = df.dropna()
df = df.drop_duplicates()

# Convert labels into numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Dataset information
print("Dataset Shape:", df.shape)
print("\nLabel Counts:")
print(df["label"].value_counts())

# Visualize spam and safe messages
sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Safe (Ham)", "Spam"])
plt.title("SpamShield AI - Message Distribution")
plt.savefig("message_distribution.png")
plt.show()

# Input and output
X = df["message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
tfidf = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel: Multinomial Naive Bayes")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Safe", "Spam"],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Spam"],
    yticklabels=["Safe", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SpamShield AI - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")