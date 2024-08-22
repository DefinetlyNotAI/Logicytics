from CODE.Custom_Libraries.Actions import Actions
output = Actions.run_command('driverquery /v')
open("Drivers.txt", "w").write(output)
