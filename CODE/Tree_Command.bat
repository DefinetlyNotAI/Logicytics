@echo off
setlocal enabledelayedexpansion

:: Define the output file name as Tree.txt
set "outputFile=Tree.txt"

:: Check if the output file already exists
if exist "!outputFile!" (
    echo WARNING: Output file already exists. Overwriting...
)

:: Run the tree command and redirect the output to the file
powershell.exe -Command "& {tree C:\ | Out-File -FilePath '!outputFile!' -Force}"

echo INFO: Completed tree command execution and saved to "!outputFile!".

endlocal
