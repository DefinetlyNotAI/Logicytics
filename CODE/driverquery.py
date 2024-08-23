from CODE.Custom_Libraries.Actions import Actions
from CODE.Custom_Libraries.Log import Log

def driverquery():
    try:
        output = Actions.run_command('driverquery /v')
        open("Drivers.txt", "w").write(output)
        Log().info("Driver Query Successful")
    except Exception as e:
        Log().error(e)
    Log().info("Driver Query Executed")