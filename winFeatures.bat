@echo off
echo INFO: Listing all enabled Windows features:
echo.

:: Get the current user's name
set USERNAME=%USERNAME%

:: Check if the DATA folder exists, if not, create it
if not exist DATA mkdir DATA

:: Create a new text file named after the user with the output of the command inside the DATA folder
dism /online /get-features /format:table > DATA\%USERNAME%_WinFeatures.txt

echo.
echo INFO: The list of enabled Windows features has been saved to DATA\%USERNAME%_WinFeatures.txt
echo.
