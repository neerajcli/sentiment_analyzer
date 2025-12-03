from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from typing import Tuple

def vectorize_data(X_train, X_test) -> Tuple:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        max_df=0.95,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()
    total_weights = X_train_vec.sum(axis=0).A1
    top_indices = total_weights.argsort()[::-1][:10]
    top_features = [feature_names[i] for i in top_indices]
    top_values = [total_weights[i] for i in top_indices]
    plt.barh(top_features[::-1], top_values[::-1])
    plt.xlabel("Total TF-IDF Weight")
    plt.title("Top 10 Features (TF-IDF)")
    plt.show()
    return X_train_vec, X_test_vec, vectorizer
