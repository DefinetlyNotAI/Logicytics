import datetime
import getpass
import os
import platform
import shutil
import subprocess
from pathlib import Path

from logicytics import check, log


def now_iso():
    return datetime.datetime.now().astimezone().isoformat()


def run_cmd(cmd):
    log.debug(f"Running command: {cmd}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            log.debug(f"Command succeeded: {cmd}")
        else:
            log.warning(f"Command returned {proc.returncode}: {cmd}")
        return proc.stdout.strip(), proc.stderr.strip(), proc.returncode
    except FileNotFoundError:
        log.error(f"Command not found: {cmd[0]}")
        return "", "not found", 127
    except subprocess.TimeoutExpired:
        log.error(f"Command timed out: {cmd}")
        return "", "timeout", 124


def have(cmd_name):
    exists = shutil.which(cmd_name) is not None
    log.debug(f"Check if '{cmd_name}' exists: {exists}")
    return exists


def get_mountvol_output():
    log.info("Gathering mounted volumes via mountvol")
    out, err, _ = run_cmd(["mountvol"])
    if not out:
        return err
    lines = out.splitlines()
    filtered = []
    keep = False
    for line in lines:
        if line.strip().startswith("\\\\?\\Volume"):
            keep = True
        if keep:
            filtered.append(line)
    return "\n".join(filtered)


def main():
    script_dir = Path(__file__).resolve().parent
    report_path = script_dir / "win_encrypted_volume_report.txt"
    log.info(f"Starting encrypted volume analysis, report will be saved to {report_path}")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Windows Encrypted Volume Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated at: {now_iso()}\n")
        f.write(f"User: {getpass.getuser()}\n")
        f.write(f"IsAdmin: {check.admin()}\n")
        f.write(f"Hostname: {platform.node()}\n")
        f.write(f"Version: {platform.platform()}\n\n")

        # Logical drives
        log.info("Gathering logical volumes via wmic")
        f.write("Logical Volumes (wmic):\n")
        out, err, _ = run_cmd(["wmic", "logicaldisk", "get",
                               "DeviceID,DriveType,FileSystem,FreeSpace,Size,VolumeName"])
        f.write(out + "\n" + err + "\n\n")

        # Mounted volumes
        f.write("Mounted Volumes (mountvol):\n")
        f.write(get_mountvol_output() + "\n\n")

        # BitLocker status
        f.write("=" * 80 + "\nBitLocker Status\n" + "=" * 80 + "\n")
        if have("manage-bde"):
            log.info("Checking BitLocker status with manage-bde")
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                path = f"{letter}:"
                if os.path.exists(f"{path}\\"):
                    out, err, _ = run_cmd(["manage-bde", "-status", path])
                    f.write(f"Drive {path}:\n{out}\n{err}\n\n")
        else:
            log.warning("manage-bde not found")

        if have("powershell"):
            log.info("Checking BitLocker status with PowerShell")
            f.write("PowerShell Get-BitLockerVolume:\n")
            ps_cmd = r"Get-BitLockerVolume | Format-List *"
            out, err, _ = run_cmd(["powershell", "-NoProfile", "-Command", ps_cmd])
            f.write(out + "\n" + err + "\n\n")
        else:
            log.warning("PowerShell not available")

    log.info(f"Report successfully saved to {report_path}")


if __name__ == "__main__":
    main()
