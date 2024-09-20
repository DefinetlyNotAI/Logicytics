import getpass
import shutil
from datetime import datetime
from __lib_class import *

log = Log(debug=DEBUG)
log_funcs = {
    "INFO": log.info,
    "WARNING": log.warning,
    "ERROR": log.error,
    "CRITICAL": log.critical,
    None: log.debug,
}


def get_default_paths():
    """Returns the default paths for photos and videos based on the Windows username."""
    username = getpass.getuser()
    default_photo_path = os.path.expanduser(f"C:\\Users\\{username}\\Pictures")
    default_video_path = os.path.expanduser(f"C:\\Users\\{username}\\Videos")
    return [default_photo_path, default_video_path]


def ensure_backup_directory_exists(backup_directory):
    """Ensures the backup directory exists; creates it if not."""
    if not os.path.exists(backup_directory):
        os.makedirs(backup_directory)


def collect_media_files(source_dirs):
    """Collects all media files from the source directories."""
    media_files = []
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png", ".mp4")):
                    media_files.append(os.path.join(root, file))
    return media_files


def backup_files(media_files, backup_directory):
    """Backs up media files to the backup directory."""
    for src_file in media_files:
        dst_file = os.path.join(
            backup_directory,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + os.path.basename(src_file),
        )
        try:
            shutil.copy2(str(src_file), str(dst_file))
            log.info(f"Copied {os.path.basename(src_file)} to {dst_file}")
        except Exception as e:
            log.error(f"Failed to copy {src_file}: {str(e)}")


def backup_media():
    """Backs up media files from the default Windows photo and video directories."""
    source_dirs = get_default_paths()
    backup_directory = "MediaBackup"
    ensure_backup_directory_exists(backup_directory)
    media_files = collect_media_files(source_dirs)
    backup_files(media_files, backup_directory)
    log.info("Media backup script completed.")


backup_media()
