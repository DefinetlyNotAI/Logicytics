from __lib_log import Log
from __lib_actions import *


def tasklist():
    try:
        result = subprocess.run(
            "tasklist /v /fo csv",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with open("tasks.csv", "wb") as file:
            file.write(result.stdout)
        Log(debug=DEBUG).info("Tasklist exported to tasks.csv")
    except subprocess.CalledProcessError as e:
        Log(debug=DEBUG).error(f"Subprocess Error: {e}")
    except Exception as e:
        Log(debug=DEBUG).error(f"Error: {e}")
