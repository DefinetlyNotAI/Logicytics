from CODE.Custom_Libraries.Actions import Actions
data = Actions.run_command("wmic BIOS get Manufacturer,Name,Version /format:htable")
open("WMIC.html", "w").write(data)
wmic_commands = [
"wmic os get Caption,CSDVersion,ServicePackMajorVersion",
"wmic computersystem get Model,Manufacturer,NumberOfProcessors",
"wmic BIOS get Manufacturer,Name,Version",
"wmic diskdrive get model,size",
]
with open("wmic_output.txt", "w") as file:
    for i in range(len(wmic_commands)):
        output = Actions.run_command(wmic_commands[i])
        file.write("-"*190)
        file.write(f"Command {i + 1}: {wmic_commands[i]}\n")
        file.write(output)
