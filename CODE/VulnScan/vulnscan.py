import joblib
import numpy as np


# Function to load and preprocess a single file for prediction
def preprocess_file(file_paths):
    with open(file_paths, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        # Adjust feature extraction to match the model's expected input shape
        feature = np.zeros(5000)  # Assuming the model expects 5000 features
        feature[0] = len(content)
        feature[1] = content.count("@")
        feature[2] = content.count("password")
        # Add more feature extraction logic here to fill the rest with the feature vector
    return [feature]  # Return as a list of features for prediction


# Function to predict if a file is sensitive
def predict_sensitive(file_paths, model_to_use):
    features = preprocess_file(file_paths)
    prediction = model_to_use.predict(features)
    confidence_score = model_to_use.predict_proba(features)[0][1]  # Confidence of the prediction (sensitive class)

    return "Not Sensitive" if prediction[0] == 1 else "Sensitive", confidence_score


# Main function to load model_to_use and make predictions
if __name__ == "__main__":
    model = joblib.load(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .2n1\trained_model.pkl")
    file_path = r"C:\Users\Hp\Desktop\Shahm\Password.txt"  # Replace with the file you want to check
    result, confidence = predict_sensitive(file_path, model)

    print(f"File '{file_path}' is classified as {result} with a confidence score of {confidence:.2f}")
    file_path = r"C:\Users\Hp\Desktop\Model Tests\giga.txt"  # Replace with the file you want to check
    result, confidence = predict_sensitive(file_path, model)

    print(f"File '{file_path}' is classified as {result} with a confidence score of {confidence:.2f}")
