1) Do checks that admin is given ✅
2) Check following flags ✅
    - --minimal
    - --basic
    - --exe (Uses exe files only)
    - --silent
    - --modded (Runs every file in the CODE directory)
    - --speedy (Only run fast commands)
    - --reboot
    - --shutdown
    - --unzip-extra (Unzips the extra directory)
    - --backup
    - --restore
    - --update
    - --extra (Opens a menu for extra dir)
    - --dev (Step to check for developers)
    - --debug
    - --webhook (Sends file via webhook if available)
3) Make sure to group flags in compulsory, and not allowed to be together etc... ✅
4) Run logicytics normally with the following things to grab - Put them in separate files)✅
   - (USE FLAGS) ✅
   - systeminfo command ✅
   - all sysinternal exe ✅
   - registry backup ✅
   - tree command ✅
   - browser data backup✅
   - windows features list✅
   - API IP Scraper✅
   - media backup ✅
   - system restore backup ❌
   - backup every file with the following names in them: password secret code login ✅
   - ssh backup ✅
   - wmic command✅
   - ipv4, ipv6, and mac address commands ✅
   - wtl file creation✅
   - log windows ✅
   - log windows bootloader and boot manager ❌
   - firewall, antivirus settings and data ❌
   - drivers used✅
   - disklist (and its subcommands) ❌
   - Property✅
   - (USE LOG CLASS FROM ALGOPY)✅
5) Make it know work all together ✅
6) In case of crashes it places the errors in ERROR.log ❌
7) Errors must be in following format ✅
    - FILECODE-ERRORCODE-FUNCTIONCODE ✅
      - Filecode: First letter of file EXCEPT _files where its first 2 letters
      - Errorcode: U -> Unknown, G -> General (exception as e), P -> Privileges error, C -> Corruption
      - Functioncode: X -> Unknown, ANY-OTHER-LETTER -> The first letter of the function, BA -> Base code, not function
    - Each code is the first letters of the name ✅
8) Zips any made data and files ✅
9) When a zip file is made, a HASH is supplied with it of the zip file✅
10) Delete all logs in event viewer ⚙️
11) Incorporate the _files fully ✅
---

1) Make .structure and .version into 1 file JSON ✅
2) [11] Make a debugger.py file that if run will attempt to debug Logicytics, ✅
   - Check:
       - date and time
       - device model
       - is it upto date with the json file in the github repo
       - python version
       - what is the path its in
       - is it in a vm
       - is it running as admin
       - execution policy
       - does it have the dependencies installed ✅
3) Add more logs to the files (debug especially) ⚙️
4) Docstring and refactor functions (:type , -> output) ✅
---

1) REDO README.
2) Add requirements.txt ✅
3) Add rules to contributing.md / remove others
   - (_files won't be seen by --mod flag)
   - (All files generated must either be directories or txt or json or md or reg or docx or png or jpeg or jpg files)
   - (Tell that libraries must have small lettered names)
   - ( Any non py file that is used MUST use special Words when printing or logging, this is due to not supporting log class custom lib)
   - [7] -> Check
4) redo the wiki
5) Add bot for the issues bug where it auto checks the logs and acts upon it
