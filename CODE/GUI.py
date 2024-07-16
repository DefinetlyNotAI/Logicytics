import subprocess  # Import the subprocess module
import tkinter as tk
from tkinter import messagebox

import mss

from local_libraries.Lists_and_variables import *

with mss.mss() as sct:
    monitors = sct.monitors
    for idx, monitor in enumerate(monitors):
        test = f"Monitor {idx}: Left={monitor['left']}, Top={monitor['top']}, Width={monitor['width']}, Height={monitor['height']}"

# Initialize the main window
root = tk.Tk()
root.title("⚠️ BETA ⚠️ Dynamic Command Builder ")
result_width = int(monitor["width"] - (monitor["width"] / 7))
result_height = int(monitor["height"] - (monitor["height"] / 7))
root.geometry(f"{result_width}x{result_height}")
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
    """
    Reset the command preview to the default state, validate the command to check for errors,
    and enable all buttons except the Execute button.
    """
    # Reset the command preview to the default state
    command_preview.set("./Logicytics.py")

    # Validate the command to check for errors
    validate_command()

    # Enable all buttons except the Execute button
    for child in buttons_grid.winfo_children():
        if isinstance(child, tk.Button) and child != execute_btn:
            child["state"] = "normal"


# Function to handle hover events and update the tooltip
def show_tooltip(event):
    """
    Shows a tooltip when the mouse hovers over a button.

    Parameters:
        event (tkinter.Event): The event object that triggered the tooltip display.

    Returns:
        None
    """
    button_text = event.widget.cget("text")
    for flag_tuple in flags:
        if flag_tuple[0] == button_text:
            tooltip_text = flag_tuple[1].strip()  # Strip whitespace
            if len(tooltip_text) > 0:  # Check if the tooltip text is not empty
                tooltip_label.config(text=tooltip_text)
                x = (
                    error_label.winfo_x()
                    + (error_label.winfo_width() / 2)
                    - (tooltip_label.winfo_width() / 2)
                )
                y = (
                    error_label.winfo_y() + error_label.winfo_height() + 10
                )  # Add some margin
                tooltip_label.place(
                    x=x, y=y
                )  # Position the tooltip below the error label
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
    """
    Hides the tooltip by clearing the tooltip text and removing the tooltip label from the window.

    This function is called when the mouse leaves an element with a tooltip.
    It clears the tooltip text by setting it to an empty string using the `config` method of the `tooltip_label` widget.
    It then removes the tooltip label from the window by calling the `pack_forget` method.
    Finally, it ensures that the tooltip label remains on top of other elements by calling the `lift` method.

    Returns:
        None
    """
    tooltip_label.config(text="")
    tooltip_label.pack_forget()
    tooltip_label.lift()  # Optionally keep it on top even when hidden


def enable_all_buttons():
    """
    Enables all the buttons by setting their state to 'normal' and then clears the disabled buttons list.
    """
    for btn in disabled_buttons:
        btn["state"] = "normal"
    disabled_buttons.clear()


# Function to validate the command and update the error label
def validate_command():
    """
    Validates the current command stored in `command_preview` against a set of rules defined by `compulsory_flags`
    and `conflicts`.
    These rules include checking for the presence of compulsory flags and ensuring there are no conflicting
    flag combinations.
    If the command fails validation, an appropriate error message is displayed, and the 'Execute' button
    is disabled to prevent execution of invalid commands.
    If the command passes validation, the error message is cleared,
    and the 'Execute' button is enabled, indicating that the command is ready to be executed.

    This function operates on global variables:
    - `command_preview`: Contains the current command to be validated.
    - `compulsory_flags`: A list of flags that must be present in the command for it to be considered valid.
    - `conflicts`: A dictionary mapping sets of conflicting flags to error messages.
    A command is considered invalid
      if it contains both flags in any set of conflicts.
    - `error_label`: A Tkinter Label widget used to display error messages.
    - `execute_btn`: A Tkinter Button widget controlling the execution of the command.
    Its state is toggled based on the
      validation result.

    Effects:
    - Modifies the text of `error_label` to reflect the validation status or error message.
    - Toggles the state of `execute_btn` between 'disabled' and 'normal' based on whether the command is valid.
    """
    full_command = command_preview.get()
    compulsory_flag_count = 0  # Initialize the counter for compulsory flags

    # Split the full command into words, excluding the initial './Logicytics.py'
    command_words = full_command.split()[1:]

    # Check if any of the compulsory flags are present in the command words
    for flag in compulsory_flags:
        # Prepare the flag for comparison by replacing underscores with hyphens and prepending '--'
        prepared_flag = "--" + flag.replace("_", "-")

        # Check if the prepared flag is in the command words
        if prepared_flag in command_words:
            compulsory_flag_count += (
                1  # Increment the counter if a compulsory flag is found
            )
            if (
                compulsory_flag_count > 1
            ):  # Check if more than one compulsory flag is found
                error_label.config(text="Error: More than one run type flag is used.")
                execute_btn["state"] = "disabled"  # Disable the Execute button
                return  # Exit the function early to prevent further execution

    # If no compulsory flags are found, set an error message accordingly
    if compulsory_flag_count == 0:
        error_label.config(text="Error: No run type flags")
        execute_btn["state"] = "disabled"  # Disable the Execute button
        return

    # Check for conflicts
    for conflict, message in conflicts.items():
        # Prepare the conflict flags for comparison
        prepared_conflict = ["--" + flag.replace("_", "-") for flag in conflict]

        # Check if both flags in the conflict are present in the command words
        if set(prepared_conflict).issubset(set(command_words)):
            error_label.config(text=message)
            execute_btn["state"] = "disabled"  # Disable the Execute button
            return

    # If no errors are found, clear the error label and enable the Execute button
    error_label.config(text="No errors")
    execute_btn["state"] = "normal"  # Enable the Execute button


def find_button_by_name(name):
    """
    Finds a button within the buttons_grid frame by its name/text.

    :param name: The name/text of the button to find.
    :return: The button widget if found, None otherwise.
    """
    for widget in buttons_grid.winfo_children():
        if isinstance(widget, tk.Button) and widget["text"] == name:
            return widget
    return None


# Modify the append_to_command_preview function to call validate_command immediately after updating the command preview
def append_to_command_preview(button_name):
    """
    Appends a button's name to the current command preview and disables the corresponding button.

    This function updates the global command preview by appending the specified button's name to it,
    effectively simulating the action of pressing the button within the context of the command line interface.
    It then locates and disables the button associated with the appended name to prevent duplicate actions.
    Finally, it calls the `validate_command()` function to automatically check the validity of the updated command.

    Parameters:
    - button_name (str): The name of the button to be appended to the command preview and disabled.

    Effects:
    - Modifies the global `command_preview`
    string by appending the new button name and joining the command parts with spaces.
    - Updates the state of the button identified by `button_name` to 'disabled',
    adding it to the `disabled_buttons` set to track disabled states.
    - Calls the `validate_command()` function to ensure the command remains valid after the update.

    Note:
        This function assumes the existence of global variables `command_preview`,
    `find_button_by_name()`, `disabled_buttons`, and `validate_command()`.
    """
    current_command = command_preview.get().split()
    current_command.append(button_name)
    command_preview.set(" ".join(current_command))

    # Find the button that was clicked and disable it
    btn = find_button_by_name(button_name)
    if btn:
        btn["state"] = "disabled"
        disabled_buttons.add(
            btn
        )  # Ensure the button is added to the disabled_buttons set

    # Automatically validate the command after updating it
    validate_command()


# Create buttons dynamically based on the flags' list
for i, (button_text, _) in enumerate(flags):
    btn = tk.Button(
        buttons_grid,
        text=button_text,
        command=lambda b=button_text: append_to_command_preview(b),
        width=20,
        height=5,
    )  # Keep the width and height as before
    btn.bind(
        "<Enter>", lambda event, b=btn: (lambda e=None: show_tooltip(e))(event), "+"
    )  # Show tooltip on hover
    btn.bind(
        "<Leave>", lambda event, b=btn: (lambda e=None: hide_tooltip())(event), "+"
    )  # Hide tooltip on mouse leave
    buttons_grid.rowconfigure(i // 4, weight=1)
    buttons_grid.columnconfigure(i % 4, weight=1)
    btn.grid(row=i // 4, column=i % 4, sticky="nsew")  # Use 'nsew' for sticky


# Function to execute the command in cmd
def execute_command(command_string):
    """
    A function that executes a command using powershell.exe based on the given command_string.

    Parameters:
    command_string (str): The command to be executed.

    Returns:
    None
    """
    try:
        command = f'powershell.exe -Command "& {command_string}"'
        subprocess.Popen(command, shell=True)
        messagebox.showinfo("Success", "Command executed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showinfo("Error", "Failed to execute the command.")


# Execute button
execute_btn = tk.Button(
    root,
    text="Execute",
    command=lambda: execute_command(command_preview.get()),
    width=20,
    height=5,
)
execute_btn.pack(side=tk.LEFT, padx=10, pady=10)
execute_btn["state"] = "disabled"  # Set the Execute button to be disabled initially

# Frame to hold the Reset button
action_buttons_frame = tk.Frame(root)
action_buttons_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Reset button
# Modify the Reset button's command to use the new function
reset_btn = tk.Button(
    action_buttons_frame,
    text="Reset",
    command=lambda: (reset_and_enable_buttons_except_execute(),),
    width=20,
    height=5,
)
reset_btn.pack(side=tk.LEFT, padx=10, pady=10)

validate_command()
root.mainloop()
