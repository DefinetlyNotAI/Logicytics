from __future__ import annotations

import os.path
from pathlib import Path


class Get:
    @staticmethod
    def list_of_files(directory: str, extensions: tuple | bool = True, append_file_list: list[str] = None,
                      exclude_files: list[str] = None) -> list:
        """
        Retrieves a list of files in the specified directory based on given extensions and exclusion criteria.
                      
        Supports two modes of file retrieval:
            1. When `extensions` is `True`, retrieves all files recursively from the directory.
            2. When `extensions` is a tuple, retrieves files matching specific extensions while applying exclusion rules.
                      
        Parameters:
            directory (str): Path of the directory to search for files.
            extensions (tuple | bool, optional): File extensions to filter or True to retrieve all files. Defaults to True.
            append_file_list (list, optional): Existing list to append found filenames to. Creates a new list if not provided. Defaults to None.
            exclude_files (list, optional): List of filenames to exclude from results. Defaults to None.
                      
        Returns:
            list: A list of filenames matching the specified criteria.
                      
        Exclusion rules:
            - Ignores files starting with an underscore (_)
            - Excludes "Logicytics.py"
            - Skips files specified in `exclude_files`
        """
        append_file_list = [] if not append_file_list else append_file_list

        if isinstance(extensions, bool) and extensions:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.relpath(os.path.join(root, filename), directory)
                    append_file_list.append(file_path)
            return append_file_list

        for filename in os.listdir(Path(directory)):
            if (
                    filename.endswith(extensions)
                    and not filename.startswith("_")
                    and filename != "Logicytics.py"
                    and (exclude_files is None or filename not in exclude_files)
            ):
                append_file_list.append(filename)
        return append_file_list

