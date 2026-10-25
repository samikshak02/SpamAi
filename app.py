import re
import joblib
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    nltk.download("stopwords", quiet=True)

    model = joblib.load("spam_pipeline.pkl")
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    return model, stop_words, stemmer


def preprocess_text(text, stop_words, stemmer):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


st.title("🛡️ SpamShield AI")
st.subheader("Intelligent SMS Spam Detection")

st.write(
    "Analyze SMS messages using Natural Language Processing "
    "and Machine Learning."
)

try:
    model, stop_words, stemmer = load_model()

except Exception:
    st.error(
        "Model files could not be loaded. "
        "Please run train_model.py first."
    )
    st.stop()


# Session history
if "history" not in st.session_state:
    st.session_state.history = []


single_tab, bulk_tab, history_tab = st.tabs([
    "📩 Single SMS",
    "📂 Bulk Detection",
    "📊 History"
])


# SINGLE SMS
with single_tab:

    st.subheader("Check Your Message")

    message = st.text_area(
        "Enter SMS message",
        placeholder="Example: Congratulations! You won a prize.",
        height=150
    )

    if st.button(
        "🔍 Analyze Message",
        type="primary"
    ):

        if not message.strip():
            st.warning("Please enter a message.")

        else:
            clean_message = preprocess_text(
                message,
                stop_words,
                stemmer
            )

            if not clean_message.strip():
                st.warning(
                    "Please enter a message containing "
                    "recognizable English text."
                )

            else:
                prediction = model.predict(
                    [clean_message]
                )[0]

                probabilities = model.predict_proba(
                    [clean_message]
                )[0]

                classes = list(model.classes_)
                spam_index = classes.index(1)

                spam_probability = (
                    probabilities[spam_index] * 100
                )

                if prediction == 1:

                    st.error("🚨 Spam Message Detected")

                else:

                    st.success("✅ Message Classified as Safe")

                st.metric(
                    "Estimated Spam Probability",
                    f"{spam_probability:.2f}%"
                )

                st.progress(
                    int(round(spam_probability))
                )

                st.caption(
                    "Probability is a model estimate, "
                    "not a guarantee that the message is safe."
                )

                st.session_state.history.append({
                    "Message": message,
                    "Prediction": (
                        "Spam" if prediction == 1
                        else "Safe"
                    ),
                    "Spam Probability (%)": round(
                        spam_probability, 2
                    )
                })


# BULK DETECTION
with bulk_tab:

    st.subheader("Analyze Multiple Messages")

    uploaded_file = st.file_uploader(
        "Upload CSV containing a 'message' column",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            if "message" not in data.columns:
                st.error(
                    "CSV must contain a column named 'message'."
                )

            elif data.empty:
                st.warning("The uploaded CSV is empty.")

            elif len(data) > 10000:
                st.warning(
                    "Please upload a CSV with at most "
                    "10,000 messages at a time."
                )

            else:
                data = data.dropna(
                    subset=["message"]
                ).copy()

                data["message"] = (
                    data["message"].astype(str)
                )

                cleaned = data["message"].apply(
                    lambda text: preprocess_text(
                        text,
                        stop_words,
                        stemmer
                    )
                )

                predictions = model.predict(
                    cleaned
                )

                probabilities = model.predict_proba(
                    cleaned
                )

                classes = list(model.classes_)
                spam_index = classes.index(1)

                data["Prediction"] = [
                    "Spam" if value == 1 else "Safe"
                    for value in predictions
                ]

                data["Spam Probability (%)"] = (
                    probabilities[:, spam_index] * 100
                ).round(2)

                st.dataframe(
                    data,
                    use_container_width=True
                )

                csv_data = data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Results",
                    data=csv_data,
                    file_name="spam_detection_results.csv",
                    mime="text/csv"
                )

        except Exception:
            st.error(
                "Unable to process this CSV. "
                "Please check its format and encoding."
            )


# HISTORY
with history_tab:

    st.subheader("Recent Predictions")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download History",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No predictions yet. Analyze an SMS to get started."
        )


st.divider()

st.caption(
    "SpamShield AI | NLP + Machine Learning | "
    "Messages are processed locally by this application."
)
