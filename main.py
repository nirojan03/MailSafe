from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = FastAPI()

model = joblib.load('spam_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

class EmailRequest(BaseModel):
    text: str

def preprocess_text(text):
    text = re.sub(r"From .*?(\n|$)", "", text)
    text = re.sub(r"Mon .*?(\n|$)", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

@app.post("/predict")
async def predict_email_spam(email: EmailRequest):
    try:
        cleaned_text = preprocess_text(email.text)
        transformed_text = tfidf_vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)
        result = "spam" if prediction[0] == 1 else "not spam"

        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))