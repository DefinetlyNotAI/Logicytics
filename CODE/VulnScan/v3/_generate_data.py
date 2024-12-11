from __future__ import annotations

import os
import random
import string
import configparser
from faker import Faker


def generate_random_filename(extensions: str, suffix_x: str = '') -> str:
    """
    Generate a random filename with the given extension and optional suffix.

    Args:
        extensions (str): The file extension.
        suffix_x (str, optional): An optional suffix to add to the filename.

    Returns:
        str: The generated random filename.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + suffix_x + extensions


def generate_content_for_extension(extensions: str, size: int | float) -> tuple[str, str]:
    """
    Generate content based on the file extension and size.

    Args:
        extensions (str): The file extension.
        size (int | float): The size of the content to generate.

    Returns:
        tuple[str, str]: The generated content and a suffix indicating the sensitivity level.
    """
    full_sensitive_chance = float(config.get('full_sensitive_chance', '0.1'))
    partial_sensitive_chance = float(config.get('partial_sensitive_chance', '0.3'))

    def generate_sensitive_data() -> str:
        """
        Generate sensitive data based on the file extension.

        Returns:
            str: The generated sensitive data.
        """
        sensitive_data_generators = {
            '.txt': lambda: random.choice([
                fake.credit_card_number(),
                fake.ssn(),
                fake.password(),
                fake.email(),
                fake.phone_number(),
                fake.iban(),
            ]),
            '.json': lambda: {
                'credit_card': fake.credit_card_number(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'password': fake.password(),
                'iban': fake.iban(),
            },
            '.csv': lambda: ",".join([
                fake.credit_card_number(),
                fake.email(),
                fake.phone_number(),
            ]),
            '.xml': lambda: f"<sensitive>{random.choice([fake.credit_card_number(), fake.iban(), fake.password()])}</sensitive>",
            '.log': lambda: f"{fake.date_time()} - Sensitive Data: {random.choice([fake.email(), fake.password(), fake.ipv4_private()])}",
            'default': lambda: fake.text(max_nb_chars=50)
        }

        return sensitive_data_generators.get(extensions, sensitive_data_generators['default'])()

    def generate_regular_content(extension_grc: str, sizes: int | float) -> str:
        """
        Generate regular content based on the file extension and size.

        Args:
            extension_grc (str): The file extension.
            sizes (int | float): The size of the content to generate.

        Returns:
            str: The generated regular content.
        """
        if extension_grc == '.txt':
            content_grc = fake.text(max_nb_chars=sizes)
        elif extension_grc == '.json':
            # noinspection PyTypeChecker
            content_grc = fake.json(data_columns={
                'name': 'name',
                'email': 'email',
                'phone': 'phone_number'
            }, num_rows=sizes // 50)
        elif extension_grc == '.csv':
            content_grc = "\n".join(
                ",".join([fake.name(), fake.email(), fake.phone_number()]) for _ in range(sizes // 50)
            )
        elif extension_grc == '.xml':
            content_grc = f"<root>{''.join([f'<item>{fake.text(50)}</item>' for _ in range(sizes // 100)])}</root>"
        elif extension_grc == '.log':
            content_grc = "\n".join([f"{fake.date_time()} - {fake.text(50)}" for _ in range(sizes // 100)])
        else:
            content_grc = fake.text(max_nb_chars=sizes)
        return content_grc

    if random.random() < full_sensitive_chance:
        if extensions == '.json':
            contents = str([generate_sensitive_data() for _ in range(size // 500)])
        elif extensions in ['.txt', '.log', '.xml']:
            contents = "\n".join(generate_sensitive_data() for _ in range(size // 500))
        elif extensions == '.csv':
            contents = "\n".join([generate_sensitive_data() for _ in range(size // 500)])
        else:
            contents = "\n".join([generate_sensitive_data() for _ in range(size // 500)])
        return contents, '-sensitive'
    else:
        regular_content = generate_regular_content(extensions, size)
        if random.random() < partial_sensitive_chance:
            sensitive_data_count = max(1, size // 500)
            sensitive_data = [generate_sensitive_data() for _ in range(sensitive_data_count)]
            regular_content_lines = regular_content.split("\n")
            for _ in range(sensitive_data_count):
                insert_position = random.randint(0, len(regular_content_lines) - 1)
                regular_content_lines.insert(insert_position, str(random.choice(sensitive_data)))
            contents = "\n".join(regular_content_lines)
            return contents, '-mix'
        else:
            contents = regular_content
            return contents, '-none'


def generate_file_content(extensions: str) -> tuple[str, str]:
    """
    Generate file content based on the file extension.

    Args:
        extensions (str): The file extension.

    Returns:
        tuple[str, str]: The generated content and a suffix indicating the sensitivity level.
    """
    size = random.randint(MIN_FILE_SIZE, MAX_FILE_SIZE)
    if SIZE_VARIATION != 0:
        variation_choice = random.choice([1, 2, 3, 4])
    if variation_choice == 1:
        size = abs(int(size + (size * SIZE_VARIATION)))
    elif variation_choice == 2:
        size = abs(int(size - (size * SIZE_VARIATION)))
    elif variation_choice == 3:
        size = abs(int(size + (size / SIZE_VARIATION)))
    elif variation_choice == 4:
        size = abs(int(size - (size / SIZE_VARIATION)))
    print(f"Generating {extensions} content of size {size} bytes")
    return generate_content_for_extension(extensions, size)


if __name__ == "__name__":
    """
    Main function to generate files based on the configuration.
    """
    fake = Faker()

    config = configparser.ConfigParser()
    config.read('../../config.ini')

    config = config['VulnScan.generate Settings']
    EXTENSIONS_ALLOWED = config.get('extensions', '.txt').split(',')
    SAVE_PATH = config.get('save_path', '.')
    CODE_NAME = config.get('code_name', 'Sense')
    SIZE_VARIATION = float(config.get('size_variation', '0.1'))

    os.makedirs(SAVE_PATH, exist_ok=True)

    DEFAULT_FILE_NUM = 10000
    DEFAULT_MIN_FILE_SIZE = 10 * 1024
    DEFAULT_MAX_FILE_SIZE = 10 * 1024

    if CODE_NAME == 'Sense':
        FILE_NUM = DEFAULT_FILE_NUM * 5
        MIN_FILE_SIZE = DEFAULT_MIN_FILE_SIZE * 5
        MAX_FILE_SIZE = DEFAULT_MAX_FILE_SIZE * 5
    elif CODE_NAME == 'SenseNano':
        FILE_NUM = 5
        MIN_FILE_SIZE = int(DEFAULT_MIN_FILE_SIZE * 0.5)
        MAX_FILE_SIZE = int(DEFAULT_MAX_FILE_SIZE * 0.5)
    elif CODE_NAME == 'SenseMacro':
        FILE_NUM = DEFAULT_FILE_NUM * 100
        MIN_FILE_SIZE = DEFAULT_MIN_FILE_SIZE
        MAX_FILE_SIZE = DEFAULT_MAX_FILE_SIZE
    elif CODE_NAME == 'SenseMini':
        FILE_NUM = DEFAULT_FILE_NUM
        MIN_FILE_SIZE = DEFAULT_MIN_FILE_SIZE
        MAX_FILE_SIZE = DEFAULT_MAX_FILE_SIZE
    else:
        MIN_FILE_SIZE = int(config['min_file_size'].replace('KB', '')) * 1024
        MAX_FILE_SIZE = int(config['max_file_size'].replace('KB', '')) * 1024
        FILE_NUM = DEFAULT_FILE_NUM

    print(f"Generating {FILE_NUM} files with sizes between {MIN_FILE_SIZE} and {MAX_FILE_SIZE} bytes")

    for i in range(FILE_NUM):
        print(f"Generating file {i + 1}/{FILE_NUM}")
        extension = random.choice(EXTENSIONS_ALLOWED).strip()
        content, suffix = generate_file_content(extension)
        filename = generate_random_filename(extension, suffix)
        filepath = os.path.join(SAVE_PATH, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Generated {FILE_NUM} files in {SAVE_PATH}")
else:
    raise ImportError("This file cannot be imported")
