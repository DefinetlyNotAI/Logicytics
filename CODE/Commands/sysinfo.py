from Custom_Libraries.Actions import *
data = Actions().run_command("systeminfo")
open("SysInfo.txt", "w").write(data)
