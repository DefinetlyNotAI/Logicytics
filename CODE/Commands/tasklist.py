import subprocess
command = 'tasklist /v /fo csv'
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
with open('tasks.csv', 'wb') as file:
    file.write(result.stdout)
