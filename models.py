import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import mask_pii

# Function to select model
def get_model(model_type: str):
    if model_type == "naive_bayes":
        return make_pipeline(TfidfVectorizer(), MultinomialNB())
    elif model_type == "svm":
        return make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    elif model_type == "decision_tree":
        return make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
    elif model_type == "random_forest":
        return make_pipeline(TfidfVectorizer(), RandomForestClassifier())
    else:
        raise ValueError("Unsupported model_type. Choose from: naive_bayes, svm, decision_tree, random_forest")

# Train and save the model using Safera dataset
def train_model(model_type: str = "svm"):
    # Dataset URL
    csv_url = "https://huggingface.co/datasets/Safera/resolve/main/combined_emails_with_natural_pii.csv"

    # Load dataset
    df = pd.read_csv(csv_url)
    if 'type' not in df.columns or 'category' not in df.columns:
        raise ValueError("CSV must contain 'type' and 'category' columns.")

    # Mask personal information
    df['masked_type'] = df['type'].apply(mask_pii)

    # Prepare training data
    emails = df['masked_type'].tolist()
    labels = df['category'].tolist()

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

    # Get and train the model
    model = get_model(model_type)
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = f"email_classifier_{model_type}.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Model '{model_type}' trained and saved to '{model_path}'.")

    return model_path
