import shutil
import os
from datetime import datetime
import getpass

def backup_media():
    # Auto-detect the Windows username
    username = getpass.getuser()

    # Default paths for photos and videos on Windows
    default_photo_path = os.path.expanduser(f'C:\\Users\\{username}\\Pictures')
    default_video_path = os.path.expanduser(f'C:\\Users\\{username}\\Videos')

    # Combine both paths for a comprehensive backup
    source_dirs = [default_photo_path, default_video_path]

    # Define the destination directory
    backup_directory = f'MediaBackup'

    if not os.path.exists(backup_directory):
        os.makedirs(backup_directory)

    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.mp4')):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(backup_directory, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + file)
                    try:
                        shutil.copy2(src_file, dst_file)
                        print(f"Copied {file} to {dst_file}")
                    except Exception as e:
                        print(f"Failed to copy {file}: {str(e)}")

# Run the backup function
backup_media()
