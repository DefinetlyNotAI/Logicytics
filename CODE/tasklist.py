import subprocess
from CODE.Custom_Libraries.Log import Log


def tasklist():
    try:
        result = subprocess.run('tasklist /v /fo csv', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open('tasks.csv', 'wb') as file:
            file.write(result.stdout)
        Log().info('Tasklist exported to tasks.csv')
    except subprocess.CalledProcessError as e:
        Log().error(f"Subprocess Error: {e}")
    except Exception as e:
        Log().error(f"Error: {e}")
