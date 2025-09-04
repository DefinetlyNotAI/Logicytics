import ctypes
import os
import winreg
from datetime import datetime, timedelta

from logicytics import log


class USBHistory:
    def __init__(self):
        self.history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usb_history.txt")

    def _save_history(self, message: str):
        """Append a timestamped message to the history file and log it."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - {message}\n"
        try:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(entry)
            log.debug(f"Saved entry: {message}")
        except Exception as e:
            log.error(f"Failed to write history: {e}")

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _get_last_write_time(root_key, sub_key_path):
        """Return the precise last write time of a registry key, or None on failure."""
        handle = ctypes.wintypes.HANDLE()
        try:
            advapi32 = ctypes.windll.advapi32
            if advapi32.RegOpenKeyExW(root_key, sub_key_path, 0, winreg.KEY_READ, ctypes.byref(handle)) != 0:
                return None
            ft = ctypes.wintypes.FILETIME()
            if advapi32.RegQueryInfoKeyW(handle, None, None, None, None, None, None, None, None, None, None,
                                         ctypes.byref(ft)) != 0:
                return None
            t = ((ft.dwHighDateTime << 32) + ft.dwLowDateTime) // 10
            return datetime(1601, 1, 1) + timedelta(microseconds=t)
        finally:
            if handle:
                ctypes.windll.advapi32.RegCloseKey(handle)

    @staticmethod
    def _enum_subkeys(root, path, warn_func):
        """Yield all subkeys of a registry key, logging warnings on errors."""
        try:
            with winreg.OpenKey(root, path) as key:
                subkey_count, _, _ = winreg.QueryInfoKey(key)
                for i in range(subkey_count):
                    try:
                        yield winreg.EnumKey(key, i)
                    except OSError as e:
                        if getattr(e, "winerror", None) == 259:  # ERROR_NO_MORE_ITEMS
                            break
                        warn_func(f"Error enumerating {path} index {i}: {e}")
        except OSError as e:
            warn_func(f"Failed to open registry key {path}: {e}")

    @staticmethod
    def _get_friendly_name(dev_info_path, device_id):
        """Return the friendly name of a device if available, else the device ID."""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dev_info_path) as dev_key:
                return winreg.QueryValueEx(dev_key, "FriendlyName")[0]
        except FileNotFoundError:
            return device_id
        except Exception as e:
            log.warning(f"Failed to read friendly name for {dev_info_path}: {e}")
            return device_id

    def read(self):
        """Read all USB devices from USBSTOR and log their info."""
        log.info("Starting USB history extraction...")
        reg_path = r"SYSTEM\CurrentControlSet\Enum\USBSTOR"
        try:
            for device_class in self._enum_subkeys(winreg.HKEY_LOCAL_MACHINE, reg_path, log.warning):
                dev_class_path = f"{reg_path}\\{device_class}"
                for device_id in self._enum_subkeys(winreg.HKEY_LOCAL_MACHINE, dev_class_path, log.warning):
                    dev_info_path = f"{dev_class_path}\\{device_id}"
                    friendly_name = self._get_friendly_name(dev_info_path, device_id)
                    last_write = self._get_last_write_time(winreg.HKEY_LOCAL_MACHINE, dev_info_path) or "Unknown"
                    self._save_history(f"USB Device Found: {friendly_name} | LastWriteTime: {last_write}")
            log.info(f"USB history extraction complete, saved to {self.history_path}")
        except Exception as e:
            log.error(f"Error during USB history extraction: {e}")


if __name__ == "__main__":
    USBHistory().read()
