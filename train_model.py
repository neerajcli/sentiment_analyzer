from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from vectorize_data import vectorize_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_vect, X_test_vect, vectorizer = vectorize_data(X_train, X_test)
    model = SVC(kernel='linear')
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.show()
    return model, vectorizer
