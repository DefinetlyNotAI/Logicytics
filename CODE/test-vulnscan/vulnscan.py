import joblib


# Function to load and preprocess a single file for prediction
def preprocess_file(file_paths):
    with open(file_paths, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        feature = len(content), content.count("@")  # Example features
    return [feature]  # Return as a list of features for prediction


# Function to predict if a file is sensitive
def predict_sensitive(file_paths, model_to_use):
    features = preprocess_file(file_paths)
    prediction = model_to_use.predict(features)
    confidence_score = model_to_use.predict_proba(features)[0][1]  # Confidence of the prediction (sensitive class)

    return "Sensitive" if prediction[0] == 1 else "Not Sensitive", confidence_score


# Main function to load model_to_use and make predictions
if __name__ == "__main__":
    model = joblib.load("trained_model.pkl")
    file_path = "file_to_check.txt"  # Replace with the file you want to check
    result, confidence = predict_sensitive(file_path, model)

    print(f"File '{file_path}' is classified as {result} with a confidence score of {confidence:.2f}")

# TODO Test the code with a file to check if it is sensitive or not
