@echo off
setlocal enabledelayedexpansion

:: Ensure the script runs with administrative privileges
>nul 2>&1 reg load HKU\TempHive C:\Windows\System32\config\systemprofile\Desktop\NTUSER.DAT

:: Redirect stdout and stderr to nul to suppress all output except echo statements
>nul 2>&1 (
    :: Check if the DATA directory exists
    if not exist DATA (
        echo Creating DATA directory...
        mkdir DATA
    )

    :: Export HKEY_CURRENT_USER registry hive
    echo Exporting HKEY_CURRENT_USER registry hive...
    reg export HKEY_CURRENT_USER "DATA\HKCU_Registry.reg" /y
)

echo INFO: REGISTRY Copied Successfully
endlocal
