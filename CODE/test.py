import requests


def send_zip_file(webhook_url, zip_path):
    with open(zip_path, "rb") as f:
        files = {"file": (f.name, f)}
    response = requests.post(webhook_url, files=files)
    return response.status_code == 200


# Usage
webhook_url = (
    "https://discordapp.com/api/webhooks/1277291355168833648/6HUNALvzz3-_-"
    "VfN2mLSp0DQCsV4DFANHVvdJmEJQQepyZyKATcsySLC-UCKh06xEYsj"
)
zip_path = "../ACCESS/DATA/Zip/backup.zip"
result = send_zip_file(webhook_url, zip_path)
print(f"File sent successfully: {result}")
