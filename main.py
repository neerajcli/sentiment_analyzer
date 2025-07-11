import tkinter as tk
from tkinter import messagebox
from train_model import train_model
from predict import predict_sentiment

model, vectorizer = train_model()

def analyze_sentiment():
    text = entry.get()
    if text:
        sentiment = predict_sentiment(model, vectorizer, text)
        messagebox.showinfo("Sentiment Analysis", f"Predicted Sentiment: {sentiment}")
    else:
        messagebox.showwarning("Input Error", "Please enter a social media post.")
root = tk.Tk()
root.title("Social Media Sentiment Analysis")
label = tk.Label(root, text="Enter a social media post:")
label.pack(pady=10)
entry = tk.Entry(root, width=50)
entry.pack(pady=10)
button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
button.pack(pady=10)
root.mainloop()
