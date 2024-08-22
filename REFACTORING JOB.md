1) Do checks that admin is given ✅
2) Check following flags ✅
    --minimal
    --basic
    --exe (Uses exe files only)
    --silent
    --modded (Runs every file in the CODE directory)
    --speedy (Only run fast commands)
        --reboot
        --shutdown

    --unzip-extra (Unzips the extra directory)
    --backup
    --restore
    --update
    --extra (Opens a menu for extra dir)
    --dev (Step to check for developers)
    --debug
    --webhook (Sends file via webhook if available)

3) Make sure to group in compulsory, and not allowed to be together ✅

4) Run logicytics normally with the following things to grab - Put them in separate files)
(USE FLAGS)
        systeminfo command ✅
        all sysinternal exe ✅
        registry backup ✅
        tree command ✅
        browser data backup
        windows features list
        API IP Scraper
                    output = (
                        f"Country: {data['location']['country']}\n"
                        f"City: {data['location']['city']}\n"
                        f"ISP: {data['network']['autonomous_system_organization']}\n"
                        f"Organization: {data['network']['autonomous_system_organization']}\n\n"
                        f"VPN Used: {'Yes' if data['security']['vpn'] else 'No'}\n"
                        f"Proxy Used: {'Yes' if data['security']['proxy'] else 'No'}\n"
                        f"Tor Used: {'Yes' if data['security']['tor'] else 'No'}\n"
                    )
                    From https://ipify.org
        media backup ✅
        system restore backup ❌
        backup everyfile with the following names in them: password secret code login
        ssh backup
        wmic command
        ipv4, ipv6, and mac address commands
        wtl file creation
        log windows bootloader and boot manager
        firewall, antivirus settings and data
        drivers used
        disklist (and its subcommands) ❌
        Property
           Property(C): WindowsBuild = 9600
           Property(C): PhysicalMemory = 16024
           Property(C): VirtualMemory = 8223
           Property(C): LogonUser = Hp
           Property(C): UserSID = S-1-5-21-3364260432-1847390784-2640483925-1001
           Property(C): UserLanguageID = 2057
           Property(C): ComputerName = SANS
           Property(C): SystemLanguageID = 2057
           Property(C): Time = 00:04:17
           Property(C): Date = 22/08/2024
           Property(C): USERNAME = HP
           Property(C): COMPANYNAME = HP
           Property(C): ROOTDRIVE = C:\
       (USE LOG CLASS FROM ALGOPY)

5) In case of crashes it places the errors in ERROR.log
6) Errors must be in following format
    FILECODE-ERRORCODE-FUNCTIONCODE
    each code is the first letters of the name
7) Zips any made data and files
8) When a zip file is made, a HASH is supplied with it of the zip file

ALSO

Make .structure and .version into 1 file JSON

Make a debugger.py file that if run will attempt to debug Logicytics,
Check:
    date and time
    device model
    is it upto date with the json file in the github repo
    python version
    what is the path its in
    is it in a vm
    is it running as admin
    execution policy
    does it have the dependencies installed

REDO README.

Add requirements.txt