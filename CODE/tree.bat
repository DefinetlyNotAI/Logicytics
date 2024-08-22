@echo off
setlocal enabledelayedexpansion
:: Define the output file name as Tree.txt
set "outputFile=Tree.txt"
:: Run the tree command and redirect the output to the file
powershell.exe -Command "& {tree C:\ | Out-File -FilePath '!outputFile!' -Force}"

endlocal