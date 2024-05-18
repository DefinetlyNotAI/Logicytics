@echo off
setlocal enabledelayedexpansion

:: Start of the script
echo Script started...

:: Assuming these variables are already set or obtained elsewhere
set "processor_architecture=x86"
set "username=JohnDoe"
set "computername=PC123"
set "number_of_processors=4"

:: Capture system info
echo Capturing system information...
systeminfo >%temp%\info.txt

:: Check if systeminfo command was successful
if exist "%temp%\info.txt" (
    echo System information captured successfully.
) else (
    echo Failed to capture system information.
    exit /b 1
)

:: Extract specific lines from the systeminfo output
echo Extracting system details...
for /f "tokens=2 delims=:" %%a in ('type %temp%\info.txt ^| find "Registered Owner"') do set owner=%%a
for /f "tokens=2 delims=:" %%a in ('type %temp%\info.txt ^| find "OS Name"') do set osname=%%a
for /f "tokens=2 delims=:" %%a in ('type %temp%\info.txt ^| find "System Manufacturer"') do set manufacture=%%a
for /f "tokens=2 delims=:" %%a in ('type %temp%\info.txt ^| find "Product ID"') do set productkey=%%a

:: Clean up the extracted data
echo Cleaning up extracted data...
del %temp%\info.txt
set owner=!owner: =!
set osname=!osname:~19!
set manufacture=!manufacture:~7!
set productkey=!productkey: =!

:: Determine architecture based on processor_architecture variable
echo Determining system architecture...
if /I "%processor_architecture%"=="x86" set arch=x32
if /I "%processor_architecture%"=="x86_64" set arch=x64
if /I "%processor_architecture%"=="AMD64" set arch=x64

:: Write the gathered information to PC_Info.txt
echo Writing system information to PC_Info.txt...
(
    echo SYSTEM: Username:%username%
    echo SYSTEM: Hostname:%computername%
    echo SYSTEM: OS:%osname%
    echo SYSTEM: Owner:%owner%
    echo SYSTEM: Product Key:%productkey%
    echo SYSTEM: Architecture:%arch%
    echo SYSTEM: Number of Processors:%number_of_processors%
) > PC_Info.txt

echo System information saved to PC_Info.txt.
echo Script completed successfully.

endlocal
