@echo off
setlocal

:: Debugging: Echo the path to MpCmdRun.exe to verify it's being resolved correctly
echo SYSTEM: Checking MpCmdRun.exe path: C:\Program Files\Windows Defender\MpCmdRun.exe

:: Check if Windows Defender signatures are removed
for /f "tokens=*" %%a in ('"C:\Program Files\Windows Defender\MpCmdRun.exe" -ShowSignatureUpdates') do (
    if "%%a"=="No signature updates are available." (
        echo INFO: Signature updates are already removed. Reinstalling now...
        "C:\Program Files\Windows Defender\MpCmdRun.exe" -UpdateSignature
    ) else (
        echo INFO: Signature updates are available. Removing now...
        "C:\Program Files\Windows Defender\MpCmdRun.exe" -RemoveDefinitions -All
    )
)

endlocal
