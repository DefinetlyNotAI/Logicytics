import tkinter as tk
from tkinter import messagebox
import subprocess  # Import the subprocess module

# Assuming Flags_Library.py contains your flags data, compulsory_flags, and conflicts
try:
    from Flags_Library import *
except ImportError:
    print("Flags_Library.py not found. Please ensure it exists and contains the necessary data.")
    exit(1)

# Initialize the main window
root = tk.Tk()
root.title("Dynamic Command Builder")
root.geometry("400x300")
disabled_buttons = set()

# Variable to hold the command preview
command_preview = tk.StringVar(root, "./Logicytics.py")

# Label to display the command preview
preview_label = tk.Label(root, textvariable=command_preview, wraplength=350)
preview_label.pack(pady=10)

# Error label
error_label = tk.Label(root, text="", fg="red")
error_label.pack(side=tk.TOP, padx=10, pady=10)  # Place above the tooltip label

# Tooltip label
tooltip_label = tk.Label(root, text="", bg="white", borderwidth=0, relief="flat")
tooltip_label.pack_forget()  # Initially hide the tooltip

# Create a grid to hold the dynamic buttons
buttons_grid = tk.Frame(root)
buttons_grid.pack(fill=tk.BOTH, expand=True)

# Configure the grid to allow expansion
buttons_grid.rowconfigure(0, weight=1)
buttons_grid.columnconfigure(0, weight=1)


def reset_and_enable_buttons_except_execute():
    # Reset the command preview to the default state
    command_preview.set("./Logicytics.py")

    # Validate the command to check for errors
    validate_command()

    # Enable all buttons except the Execute button
    for child in buttons_grid.winfo_children():
        if isinstance(child, tk.Button) and child != execute_btn:
            child['state'] = 'normal'


# Function to handle hover events and update the tooltip
def show_tooltip(event):
    button_text = event.widget.cget("text")
    for flag_tuple in flags:
        if flag_tuple[0] == button_text:
            tooltip_text = flag_tuple[1].strip()  # Strip whitespace
            if len(tooltip_text) > 0:  # Check if the tooltip text is not empty
                tooltip_label.config(text=tooltip_text)
                x = error_label.winfo_x() + (error_label.winfo_width() / 2) - (tooltip_label.winfo_width() / 2)
                y = error_label.winfo_y() + error_label.winfo_height() + 10  # Add some margin
                tooltip_label.place(x=x, y=y)  # Position the tooltip below the error label
                tooltip_label.lift()  # Ensure the tooltip is always on top
                break
            else:
                tooltip_label.config(text="")  # Clear the tooltip text if it's empty
                tooltip_label.place_forget()  # Remove the tooltip label from the window
                break
    else:
        tooltip_label.config(text="")
        tooltip_label.place_forget()  # Hide the tooltip if no matching flag is found


# Function to handle mouse leave events to hide the tooltip
def hide_tooltip():
    tooltip_label.config(text="")
    tooltip_label.pack_forget()
    tooltip_label.lift()  # Optionally keep it on top even when hidden


def enable_all_buttons():
    for btn in disabled_buttons:
        btn['state'] = 'normal'
    disabled_buttons.clear()


# Function to validate the command and update the error label
def validate_command():
    full_command = command_preview.get()
    compulsory_flag_count = 0  # Initialize the counter for compulsory flags

    # Split the full command into words, excluding the initial './Logicytics.py'
    command_words = full_command.split()[1:]

    # Check if any of the compulsory flags are present in the command words
    for flag in compulsory_flags:
        # Prepare the flag for comparison by replacing underscores with hyphens and prepending '--'
        prepared_flag = '--' + flag.replace('_', '-')

        # Check if the prepared flag is in the command words
        if prepared_flag in command_words:
            compulsory_flag_count += 1  # Increment the counter if a compulsory flag is found
            if compulsory_flag_count > 1:  # Check if more than one compulsory flag is found
                error_label.config(text="Error: More than one run type flag is used.")
                execute_btn['state'] = 'disabled'  # Disable the Execute button
                return  # Exit the function early to prevent further execution

    # If no compulsory flags are found, set an error message accordingly
    if compulsory_flag_count == 0:
        error_label.config(text="Error: No run type flags")
        execute_btn['state'] = 'disabled'  # Disable the Execute button
        return

    # Check for conflicts
    for conflict, message in conflicts.items():
        # Prepare the conflict flags for comparison
        prepared_conflict = ['--' + flag.replace('_', '-') for flag in conflict]

        # Check if both flags in the conflict are present in the command words
        if set(prepared_conflict).issubset(set(command_words)):
            error_label.config(text=message)
            execute_btn['state'] = 'disabled'  # Disable the Execute button
            return

    # If no errors are found, clear the error label and enable the Execute button
    error_label.config(text="No errors")
    execute_btn['state'] = 'normal'  # Enable the Execute button


def find_button_by_name(name):
    """
    Finds a button within the buttons_grid frame by its name/text.

    :param name: The name/text of the button to find.
    :return: The button widget if found, None otherwise.
    """
    for widget in buttons_grid.winfo_children():
        if isinstance(widget, tk.Button) and widget['text'] == name:
            return widget
    return None


# Modify the append_to_command_preview function to call validate_command immediately after updating the command preview
def append_to_command_preview(button_name):
    current_command = command_preview.get().split()
    current_command.append(button_name)
    command_preview.set(' '.join(current_command))

    # Find the button that was clicked and disable it
    btn = find_button_by_name(button_name)
    if btn:
        btn['state'] = 'disabled'
        disabled_buttons.add(btn)  # Ensure the button is added to the disabled_buttons set

    # Automatically validate the command after updating it
    validate_command()


# Create buttons dynamically based on the flags' list
for i, (button_text, _) in enumerate(flags):
    btn = tk.Button(buttons_grid, text=button_text, command=lambda b=button_text: append_to_command_preview(b),
                    width=20, height=5)  # Keep the width and height as before
    btn.bind("<Enter>", lambda event, b=btn: (lambda e=None: show_tooltip(e))(event), "+")  # Show tooltip on hover
    btn.bind("<Leave>", lambda event, b=btn: (lambda e=None: hide_tooltip())(event),
             "+")  # Hide tooltip on mouse leave
    buttons_grid.rowconfigure(i // 4, weight=1)
    buttons_grid.columnconfigure(i % 4, weight=1)
    btn.grid(row=i // 4, column=i % 4, sticky='nsew')  # Use 'nsew' for sticky


# Function to execute the command in cmd
def execute_command(command_string):
    try:
        command = f'powershell.exe -Command "& {command_string}"'
        subprocess.Popen(command, shell=True)
        messagebox.showinfo("Success", "Command executed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showinfo("Error", "Failed to execute the command.")


# Execute button
execute_btn = tk.Button(root, text="Execute", command=lambda: execute_command(command_preview.get()), width=20,
                        height=5)
execute_btn.pack(side=tk.LEFT, padx=10, pady=10)
execute_btn['state'] = 'disabled'  # Set the Execute button to be disabled initially

# Frame to hold the Reset button
action_buttons_frame = tk.Frame(root)
action_buttons_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Reset button
# Modify the Reset button's command to use the new function
reset_btn = tk.Button(action_buttons_frame, text="Reset",
                      command=lambda: (reset_and_enable_buttons_except_execute(),), width=20,
                      height=5)
reset_btn.pack(side=tk.LEFT, padx=10, pady=10)

validate_command()
root.mainloop()
