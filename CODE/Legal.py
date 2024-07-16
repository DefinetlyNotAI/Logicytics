import os
import shutil
import tkinter as tk
from tkinter import messagebox  # Explicitly import messagebox

# Initialize the main window
root = tk.Tk()
root.title("Logicytics Consent")

# Variable to track the state of the Fun Summary/Legal Words button
fun_summary_state = tk.BooleanVar(value=True)

# List to hold the initial texts
initial_texts = [
    """
TERMS OF SERVICE

### Title:
"Welcome to Logicytics Let's Get This Party Started (Legally)."

### Content:

**Terms and Services Agreement:**
- By using Logicytics, you agree to abide by our Terms and Conditions. We promise not to share your data with anyone who doesn't know how to keep a secret. 🤫

**Legal Realm Compliance:**
- Please note, any legal issues arising from the use of Logicytics are null and void under the following conditions:
  - **Version Breach:** You are using a version of Logicytics that is not the latest. Remember, staying updated is like getting a new toy every day. Who wouldn't love that? 🎁
  - **External Modification:** You have modified Logicytics externally. While we appreciate creativity, sticking to the original recipe ensures the best results. 🍲

**Data Forensics & System Data Mining:**
- At Logicytics, we believe in the power of data. Our tools are designed for data forensics and system data mining, helping you uncover insights that were previously hidden. Think of us as detectives, but with way cooler gadgets. 🔍🔬

**Humor & Light-hearted Tone:**
- We take your privacy seriously, but not ourselves too much. After all, life's too short for boring legal jargon. So, enjoy your journey with Logicytics, where data meets fun. 😉

**Disclaimer:**
- Nothing shared through Logicytics leaves the digital realm. It's like having a private conversation in a crowded room—no one else hears a thing. 👂""",
    """
TERMS OF SERVICE

### Popup Title:
"Logicytics User Agreement and Privacy Notice"

### Content:

**User Agreement:**
- By accessing and using Logicytics, you agree to comply with the following terms and conditions. Your continued use signifies your acceptance of these policies.

**Privacy Notice:**
- Logicytics values your privacy. We collect minimal personal data necessary for the operation of our services and commit to protecting it according to applicable laws and regulations.

**Legal Considerations:**
- Please be aware that any legal disputes arising from the use of Logicytics are subject to the jurisdiction of the courts located in [Jurisdiction]. Use of outdated versions or unauthorized modifications may invalidate your rights under this agreement.

**Data Handling:**
- Logicytics is dedicated to data forensics and system data mining, ensuring that all operations are conducted within legal boundaries. No data collected or processed through our platform is shared online or with third parties without your explicit consent.

**Important Legal Provisions:**
- Violation of the following provisions will result in immediate termination of access and may lead to legal action:
  - Using a non-latest version of Logicytics.
  - Engaging in external modifications to the software.

**Disclaimers:**
- Logicytics disclaims liability for any damages resulting from the use of its services. Users assume full responsibility for their actions and interpretations of data obtained through our platform.""",
]

# Variable to track the current index of the initial text
current_initial_text_index = 0


# Function to check for existing.accept files


def check_for_existing_accept_files() -> bool:
    """
    Check if there is any file ending with '.accept' in the current directory.

    Returns:
        True if a file ending with '.accept' is found, False otherwise.
    """
    # Get the parent directory of the current working directory
    finalpath = os.path.dirname(os.getcwd())

    # Iterate over the files in the SYSTEM directory
    for filename in os.listdir(os.path.join(finalpath, "SYSTEM")):
        # Check if the file ends with '.accept'
        if filename.endswith(".accept"):
            return True

    # If no file ending with '.accept' is found, return False
    return False


# Perform the check immediately upon opening
if check_for_existing_accept_files():
    print("Found ToS agreement file, proceeding.")
    root.destroy()
else:
    # Display the first initial text
    text_label = tk.Label(root, text=initial_texts[current_initial_text_index])
    text_label.pack(pady=20)  # Use pack for simpler layout

    def on_accept_click():
        """
        Handles the click event when the user confirms their agreement to the terms.
        Creates a file with the agreement confirmation, moves it to a different directory,
        and deletes the original file.
        """
        # Confirm the user's agreement
        agreement_confirmation = messagebox.askyesno(
            title="Agreement Confirmation",
            message="Do you confirm that you agree to the terms?",
            parent=root,
        )
        try:
            if agreement_confirmation:
                # Create a file with the agreement confirmation
                with open("ToS.accept", "w") as f:
                    f.write("You have agreed to the terms.\n")

                # Inform the user that the file has been created
                messagebox.showinfo(
                    title="Success", message="A file 'ToS.accept' has been created."
                )

                # Construct the source path of the ToS.accept file
                current_working_dir = os.getcwd()
                source_path = os.path.join(current_working_dir, "ToS.accept")

                # Check if the file exists at the source path
                if os.path.exists(source_path):
                    # Construct the destination path in the parent directory and then in the SYSTEM directory
                    parent_dir = os.path.dirname(current_working_dir)
                    system_dir = os.path.join(parent_dir, "SYSTEM")
                    destination_path = os.path.join(system_dir, "ToS.accept")

                    # Ensure the SYSTEM directory exists; otherwise, create it
                    os.makedirs(system_dir, exist_ok=True)

                    # Move the file to the destination path
                    shutil.move(source_path, destination_path)

                    # Delete the original file
                    os.remove(source_path)

                    print(
                        f"File moved successfully to {destination_path}. Original file deleted."
                    )
                else:
                    print(
                        "The file ToS.accept does not exist in the current directory."
                    )

                # Close the application
                root.destroy()

            else:
                messagebox.showwarning(
                    title="Warning",
                    message="Agreement not confirmed. Application remains open.",
                )
        except Exception:
            # Close the application
            root.destroy()

    def on_reject_click():
        """
        Handles the event when the user rejects the terms.
        Closes the application window.
        """
        # Close the application
        root.destroy()

    def on_fun_summary_click():
        """
        This function handles the click event for the fun summary/legal words button.
        Toggles the state of the button and updates the button text accordingly.
        Also toggle the initial text displayed on the interface.
        """
        global current_initial_text_index  # Declare it as global if necessary

        # Toggle the state of the fun summary/legal words button
        fun_summary_state.set(not fun_summary_state.get())

        if fun_summary_state.get():
            fun_summary_button.config(text="Click Here for Legal Words")
        else:
            fun_summary_button.config(text="Click Here for Fun Words")

        # Toggle the initial text
        current_initial_text_index = (current_initial_text_index + 1) % len(
            initial_texts
        )
        text_label.config(text=initial_texts[current_initial_text_index])

    # Accept button
    accept_button = tk.Button(root, text="Accept", command=on_accept_click)
    accept_button.pack(pady=10)  # Use pack for simpler layout

    # Reject button
    reject_button = tk.Button(root, text="Reject", command=on_reject_click)
    reject_button.pack(pady=10)  # Use pack for simpler layout

    # Fun Summary/Legal Words button
    fun_summary_button = tk.Button(
        root, text="Click Here for Legal Words", command=on_fun_summary_click
    )
    fun_summary_button.pack(pady=10)  # Use pack for simpler layout

# Start the application
root.mainloop()
