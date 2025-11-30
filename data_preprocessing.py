import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    df = pd.read_csv('social_media_sentiment.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df['label'].value_counts().plot(kind='bar', title="Class Distribution", color=['green', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()
    X = df['post']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

