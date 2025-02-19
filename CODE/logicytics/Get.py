from __future__ import annotations

import os


class Get:
    @staticmethod
    def list_of_files(
            directory: str,
            only_extensions: list[str] = None,
            append_file_list: list[str] = None,
            exclude_files: list[str] = None,
            exclude_extensions: list[str] = None,
    ) -> list[str]:
        """
        Retrieves a list of files in the specified directory based on given extensions and exclusion criteria.

        Parameters:
            directory (str): Path of the directory to search for files.
            only_extensions (list[str], optional): List of file extensions to filter. If None, retrieves all files. Defaults to None.
            append_file_list (list[str], optional): Existing list to append found filenames to. Defaults to None.
            exclude_files (list[str], optional): List of filenames to exclude from results. Defaults to None.
            exclude_extensions (list[str], optional): List of extensions to exclude from results. Defaults to None.

        Returns:
            list[str]: A list of filenames matching the specified criteria.

        Exclusion rules:
            - Ignores files starting with an underscore (_)
            - Skips files specified in `exclude_files`
        """
        append_file_list = append_file_list or []
        exclude_files = set(exclude_files or [])
        exclude_extensions = set(exclude_extensions or [])

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.startswith("_") or filename in exclude_files:
                    continue  # Skip excluded files
                if any(filename.endswith(ext) for ext in exclude_extensions):
                    continue  # Skip excluded files

                file_path = os.path.relpath(os.path.join(root, filename), directory)

                if only_extensions is None or any(filename.endswith(ext) for ext in only_extensions):
                    append_file_list.append(file_path)

        return append_file_list
