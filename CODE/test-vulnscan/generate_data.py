import os
import random

from faker import Faker

# Initialize the Faker instance
fake = Faker()

# Maximum file size in bytes (default to 1 GB)
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB


# Function to generate a sensitive file with real sensitive information
def create_sensitive_file(file_path, max_size=MAX_FILE_SIZE):
    content = ""
    # Generate sensitive data using Faker
    content += f"Name: {fake.name()}\n"
    content += f"Address: {fake.address()}\n"
    content += f"Phone: {fake.phone_number()}\n"
    content += f"Email: {fake.email()}\n"
    content += f"Credit Card: {fake.credit_card_number()}\n"
    content += f"SSN: {fake.ssn()}\n"
    content += f"Company: {fake.company()}\n"

    # Keep adding more sensitive data until the file size is less than the max limit
    while len(content.encode('utf-8')) < max_size:
        content += f"Sensitive Info: {fake.text(max_nb_chars=200)}\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Function to generate a normal file with non-sensitive data
def create_normal_file(file_path, max_size=MAX_FILE_SIZE):
    content = ""
    # Add random text
    while len(content.encode('utf-8')) < max_size:
        content += fake.text(max_nb_chars=200) + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Function to generate a mix file with both normal and sensitive data
def create_mix_file(file_path, max_size=MAX_FILE_SIZE):
    content = ""
    # Add a mix of normal and sensitive data
    while len(content.encode('utf-8')) < max_size:
        if random.choice([True, False]):
            content += fake.text(max_nb_chars=200) + "\n"  # Normal data
        else:
            content += f"Name: {fake.name()}\n"
            content += f"Credit Card: {fake.credit_card_number()}\n"
            content += f"SSN: {fake.ssn()}\n"  # Sensitive data

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Function to create random files (Normal, Mix, Sensitive)
def create_random_files(directories, num_file=100):
    os.makedirs(directories, exist_ok=True)

    for i in range(num_file):
        file_type = random.choice(['Normal', 'Mix', 'Sensitive'])
        file_name = f"file_{i + 1}_{file_type}.txt"
        file_path = os.path.join(directories, file_name)

        if file_type == "Sensitive":
            create_sensitive_file(file_path, MAX_FILE_SIZE)
        elif file_type == "Mix":
            create_mix_file(file_path, MAX_FILE_SIZE)
        else:
            create_normal_file(file_path, MAX_FILE_SIZE)

        print(f"Created {file_type} file: {file_name}")


# Main function to call the file creation logic
if __name__ == "__main__":
    directory = "generated_data_v2"
    num_files = 50000  # Adjust as needed
    MAX_FILE_SIZE = 20 * 1024  # Example: Max file size is 20 KB

    create_random_files(directory, num_file=num_files)


# TODO Fix the code to generate the data for the vulnscan project
# Epoch 1/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_1.pkl
# Epoch 2/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_2.pkl
# Epoch 3/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_3.pkl
# Epoch 4/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_4.pkl
# Epoch 5/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_5.pkl
# Epoch 6/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_6.pkl
# Epoch 7/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_7.pkl
# Epoch 8/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_8.pkl
# Epoch 9/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_9.pkl
# Epoch 10/10
# Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1-Score: 1.00, ROC-AUC: 1.00
# Model checkpoint saved: trained_model_epoch_10.pkl
# Final model saved as trained_model.pkl
# Training complete.