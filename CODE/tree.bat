@echo off

echo INFO: Starting Tree Command

setlocal enabledelayedexpansion
:: Define the output file name as Tree.txt
set "outputFile=Tree.txt"
:: Run the tree command and redirect the output to the file
powershell.exe -Command "& {tree /f C:\ | Out-File -FilePath '!outputFile!' -Force}"

echo INFO: Saved !outputFile!

endlocal
