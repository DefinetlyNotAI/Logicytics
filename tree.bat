@echo off
setlocal

:: Get the current user's name
for /f "tokens=*" %%i in ('echo %username%') do set userName=%%i

:: Define the output file name based on the current user's name
set outputFile=%userName%_tree.txt

:: Run the tree command and redirect the output to the file
powershell.exe -Command "& {tree C:\ | Out-File -FilePath %outputFile%}"
