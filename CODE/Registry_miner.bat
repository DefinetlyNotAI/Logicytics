@echo off
setlocal

:: Check if the DATA directory exists
if not exist DATA (
    echo Creating DATA directory...
    mkdir DATA
)

:: Export HKEY_CURRENT_USER registry hive
echo Exporting HKEY_CURRENT_USER registry hive...
reg export HKEY_CURRENT_USER "DATA\HKCU_Registry.reg" /y

echo Done.
endlocal
