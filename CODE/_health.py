import os
import shutil
import zipfile
import subprocess


def backup(directory, name):
    # Check if backup exists, delete it if so
    if os.path.exists('../Access/Backup/backup.zip'):
        os.remove('../Access/Backup/backup.zip')

    # Zip the directory and move it to the backup location
    with zipfile.ZipFile(f'{name}.zip', 'w') as zip_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(str(file_path), start=os.getcwd())
                zip_file.write(str(file_path), arcname=relative_path)

    shutil.move('backup.zip', '../Access/Backup')


def update():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    subprocess.run(['git', 'pull'])
    os.chdir(current_dir)
