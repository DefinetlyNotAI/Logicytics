# Special wrapper that the exe is made out of
import subprocess
import sys


FLAG = tuple(sys.argv[1:])

if len(FLAG) == 0:
    subprocess.run(['python', 'Logicytics.py'], shell=True)

elif len(FLAG) == 2:
    flag1, flag2 = FLAG
    flag1 = '--' + flag1
    flag2 = '--' + flag2
    subprocess.run(['python', 'Logicytics.py', flag1, flag2], shell=True)

else:
    flag1 = '--' + sys.argv[1]
    subprocess.run(['python', 'Logicytics.py', flag1], shell=True)
