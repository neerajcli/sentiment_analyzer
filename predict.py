def predict_sentiment(model, vectorizer, text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]
