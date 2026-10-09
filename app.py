import gradio as gr
import joblib

# Load the trained model and text vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


def check_email(email):
    if not email or not email.strip():
        return "Please enter an email message."

    email_features = tfidf.transform([email])
    prediction = model.predict(email_features)[0]

    if prediction == "spam":
        return "🚨 SPAM: This message may be unwanted."
    else:
        return "✅ HAM: This message appears legitimate."


app = gr.Interface(
    fn=check_email,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste your email message here...",
        label="Email Message"
    ),
    outputs=gr.Textbox(label="Classification Result"),
    title="Spam Email Classifier",
    description="Enter an email message to check whether it is spam or legitimate."
)

import os

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
