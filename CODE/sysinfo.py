from CODE.Custom_Libraries.Actions import Actions
data = Actions.run_command("systeminfo")
open("SysInfo.txt", "w").write(data)
