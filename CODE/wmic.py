from logicytics import log, Execute


@log.function
def wmic():
    """
    Retrieves system information using WMIC commands.

    This function runs a series of WMIC commands to gather information about the system's BIOS,
    operating system, computer system, and disk drives.
    The output of each command is written to a file named "wmic_output.txt".

    Parameters:
    None

    Returns:
    None
    """
    data = Execute.command("wmic BIOS get Manufacturer,Name,Version /format:htable")
    with open("WMIC.html", "w") as file:
        file.write(data)
    wmic_commands = [
        "wmic os get Caption,CSDVersion,ServicePackMajorVersion",
        "wmic computersystem get Model,Manufacturer,NumberOfProcessors",
        "wmic BIOS get Manufacturer,Name,Version",
        "wmic diskdrive get model,size",
    ]
    with open("wmic_output.txt", "w") as file:
        for index, command in enumerate(wmic_commands):
            log.info(f"Executing Command Number {index + 1}: {command}")
            output = Execute.command(command)
            file.write("-" * 190)
            file.write(f"Command {index + 1}: {command}\n")
            file.write(output)

        file.write("-" * 190)


if __name__ == "__main__":
    wmic()
