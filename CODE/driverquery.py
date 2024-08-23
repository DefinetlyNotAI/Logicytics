from CODE.Custom_Libraries.Actions import *
from CODE.Custom_Libraries.Log import Log


def driverquery():
    try:
        output = Actions.run_command("driverquery /v")
        open("Drivers.txt", "w").write(output)
        Log(debug=DEBUG).info("Driver Query Successful")
    except Exception as e:
        Log(debug=DEBUG).error(e)
    Log(debug=DEBUG).info("Driver Query Executed")
