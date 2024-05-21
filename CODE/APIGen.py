import os
import tkinter as tk
from tkinter import messagebox  # Explicitly import messagebox

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
instruction_label.pack(pady=20)  # Use pack for simpler layout

# Entry widget for the user to input the API key
api_key_entry = tk.Entry(root)
api_key_entry.pack(pady=10)  # Use pack for simpler layout

# Entry widget for the user to re-enter the API key for double-entry validation
api_key_entry_confirm = tk.Entry(root)
api_key_entry_confirm.pack(pady=10)  # Use pack for simpler layout


def submit_api_key():
    api_key = api_key_entry.get().strip()  # Retrieve and strip whitespace from the entered API key
    api_key_confirm = api_key_entry_confirm.get().strip()  # Retrieve and strip whitespace from the confirmed API key

    # Error check for empty inputs
    if not api_key or not api_key_confirm:
        messagebox.showerror(title="Error", message="Both fields must be filled out.")
        return

    # Double-entry validation
    if api_key != api_key_confirm:
        messagebox.showwarning(title="Warning", message="The API keys do not match. Please try again.")
        return

    # Check if the API.KEY file already exists
    if os.path.exists('SYSTEM/API.KEY'):
        messagebox.showerror(title="Error",
                             message="A API.KEY file already exists in the SYSTEM directory. Please delete it before submitting a new API key.")
        return

    # Proceed to create the API.KEY file with the submitted API key
    parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
    system_dir = os.path.join(parent_dir, "SYSTEM")  # Construct the SYSTEM directory path
    os.makedirs(system_dir, exist_ok=True)  # Ensure the SYSTEM directory exists

    with open(os.path.join(system_dir, 'API.KEY'), 'w') as f:
        f.write(api_key + "\n")  # Write the API key to the file followed by a newline character

    messagebox.showinfo(title="Success", message="API key saved to API.KEY.")
    exit(1)


# Submit button for the user to finalize the API key submission
submit_button = tk.Button(root, text="Submit", command=submit_api_key)
submit_button.pack(pady=10)  # Use pack for simpler layout

# Start the application
root.mainloop()
