from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()
    total_features = X_train_vect.sum(axis=0).A1
    top_features = sorted(zip(total_features, feature_names), reverse=True)[:10]
    plt.barh([x[1] for x in top_features], [x[0] for x in top_features], color='blue')
    plt.xlabel('Frequency')
    plt.title('Top 10 Features (TF-IDF)')
    plt.show()
    return X_train_vect, X_test_vect, vectorizer
