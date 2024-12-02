import os
import random

from faker import Faker

MAX_FILE_SIZE = 10 * 1024  # Example: Max file size is 10 KB

# Initialize the Faker instance
fake = Faker()


# Function to generate a sensitive file with real sensitive information
def create_sensitive_file(file_path, max_size):
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
def create_normal_file(file_path, max_size):
    content = ""
    # Add random text
    while len(content.encode('utf-8')) < max_size:
        content += fake.text(max_nb_chars=200) + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Function to generate a mix file with both normal and sensitive data
def create_mix_file(file_path, max_size):
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
    directory = "generated_data_1m-files_10KB"

    create_random_files(directory, num_file=1000000)
