import os
import tkinter as tk
from tkinter import messagebox


def submit_api_key():
    """
    Submit the API key entered by the user and save it to the SYSTEM/API-IP.key file.
    """
    api_key = api_key_entry.get().strip()  # Get the API key entered by the user
    api_key_confirm = (
        api_key_entry_confirm.get().strip()
    )  # Get the API key confirmation entered by the user

    # Enhanced error check for empty inputs
    if not api_key or not api_key_confirm:
        messagebox.showerror(title="Error", message="Both fields must be filled out.")
        return

    # Validate the API key format (example: minimum length of 32 characters)
    if len(api_key) < 32:
        messagebox.showerror(
            title="Error", message="API key must be at least 32 characters long."
        )
        return

    # Double-entry validation
    if api_key != api_key_confirm:
        messagebox.showwarning(
            title="Warning", message="The API keys do not match. Please try again."
        )
        return

    # Check if the API-IP.key file already exists and read its content
    if os.path.exists("SYSTEM/API-IP.key"):
        try:
            with open("SYSTEM/API-IP.key", "r") as f:
                existing_key = f.read().strip()
            if existing_key == api_key:
                messagebox.showinfo(
                    title="Info",
                    message="The same API key is already used. No action needed.",
                )
                return
        except Exception as e:
            messagebox.showerror(
                title="Error", message=f"Failed to read existing API key: {str(e)}"
            )
            return

    # Proceed to create/update the API-IP.key file with the submitted API key
    try:
        parent_dir = os.path.dirname(
            os.getcwd()
        )  # Get the parent directory of the current working directory
        system_dir = os.path.join(
            parent_dir, "SYSTEM"
        )  # Join the parent directory and "SYSTEM" to get the system directory
        os.makedirs(
            system_dir, exist_ok=True
        )  # Create the system directory if it doesn't exist

        with open(os.path.join(system_dir, "API-IP.key"), "w") as f:
            f.write(api_key + "\n")  # Write the API key to the API-IP.key file

        messagebox.showinfo(title="Success", message="API key saved to API-IP.key.")
    except Exception as e:
        messagebox.showerror(title="Error", message=f"Failed to save API key: {str(e)}")

    root.destroy()


# Initialize the main window
root = tk.Tk()
root.title("VPN API Key Request")

# Instructions for obtaining the API key
instructions = """
Please visit https://vpnapi.io/dashboard to create an account and obtain your API key. Once you have your API key,
please enter it below and click 'Submit'.

We do apologise but for IP analysis we had to use this method, we ensure you its safe, if you are still in doubt you may use this pre-generated API key {c6048787b83f44b18d4ce50e5c8869ed}

The KEY should not include the {} given, the key has a limit of 1000 requests a day, so its recommended to use your own API key, Thank you and we apologise for this inconvenience, to skip the API function type API-NO as your key.
"""

# Label to display instructions
instruction_label = tk.Label(root, text=instructions)
instruction_label.pack(pady=20)

# Entry widgets for the user to input the API key
api_key_entry = tk.Entry(root)
api_key_entry.pack(pady=10)

# Entry widget for the user to re-enter the API key for double-entry validation
api_key_entry_confirm = tk.Entry(root)
api_key_entry_confirm.pack(pady=10)

# Submit button for the user to finalize the API key submission
submit_button = tk.Button(root, text="Submit", command=submit_api_key)
submit_button.pack(pady=10)

# Start the application
root.mainloop()
