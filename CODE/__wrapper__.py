# Special wrapper that the exe Logicytics is made out of, not compulsory, just to ignore some restrictions of python.

# If you modify please run this command:
# pyinstaller --noconfirm --onefile --console --icon "C:\Users\Hp\Desktop\Logicytics\IMG\EXE.ico"  "C:\Users\Hp\Desktop\Logicytics\CODE\__wrapper__.py"
# Assuming Logicytics is in the Desktop, and the paths are unchanged (You may need to replace Hp with your username.
# Then rename from __wrapper__.exe to Logicytics.exe

import subprocess
import sys


FLAG = tuple(sys.argv[1:])

if len(FLAG) == 0:
    subprocess.run(["python", "Logicytics.py"], shell=True)

elif len(FLAG) == 2:
    flag1, flag2 = FLAG
    subprocess.run(["python", "Logicytics.py", flag1, flag2], shell=True)

else:
    flag1 = sys.argv[1]
    subprocess.run(["python", "Logicytics.py", flag1], shell=True)
