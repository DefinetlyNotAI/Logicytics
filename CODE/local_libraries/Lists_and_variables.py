# TODO*/ Flag Setups Occur Here only if new flags are made.

# Definitions and types of flags taken raw from the shell.
flags = [
    (
        "--legacy",
        "Runs only the legacy scripts and required scripts that are made from the stable v1 project.",
    ),
    (
        "--unzip-extra",
        "Unzips all the extra files in the EXTRA directory == ONLY DO THIS ON YOUR OWN MACHINE, MIGHT TRIGGER ANTIVIRUS ==.",
    ),
    (
        "--backup",
        "Creates a backup of all the files in the CODE directory in a ZIP file in a new BACKUP Directory, == ONLY DO THIS ON YOUR OWN MACHINE ==",
    ),
    (
        "--restore",
        "Restores all files from the BACKUP directory, == ONLY DO THIS ON YOUR OWN MACHINE ==",
    ),
    (
        "--update",
        "Updates from the latest stable version in the GitHub repository, == ONLY DO THIS ON YOUR OWN MACHINE ==",
    ),
    (
        "--debug",
        "All Variables/Lists in the main project only are displayed with a DEBUG tag.",
    ),
    (
        "--extra",
        "Opens a menu for the EXTRA directory files == USE ON YOUR OWN RISK ==.",
    ),
    ("--onlypy", "Runs only the python scripts and required scripts."),
    ("--setup-only", "Runs all prerequisites then quits."),
    ("--setup", "Runs all prerequisites then Logicytics normally."),
    (
        "--minimum",
        "Runs the bare minimum where no external API or files are used, as well as running only quick programs.",
    ),
    ("--only-native", "Only runs PowerShell and Batch plus clean-up and setup script."),
    ("--debugger-only", "Runs the debugger then quits."),
    ("--debugger", "Runs the debugger then Logicytics."),
    ("--run", "Runs with default settings."),
    ("--mini-log", "Runs the log without feedback from the software."),
    ("--silent", "Runs without showing any log"),
    ("--shutdown", "After completing, ejects disk then shuts down the entire system."),
    ("--reboot", "After completing, ejects disk then restarts the entire system."),
    (
        "--bios",
        "After completing, ejects disk then restarts the entire system with instructions for you to follow.",
    ),
    (
        "--mods",
        "Runs every file in the CODE directory, useful for testing out locally developed scripts.",
    ),
    (
        "--dev",
        "Runs scripts that are only used for development purposes. These are for contributors only, these include registering new files",
    ),
]

# Flags that are compulsory to be used only once.
compulsory_flags = [
    "onlypy",
    "setup_only",
    "setup",
    "minimum",
    "only_native",
    "debugger_only",
    "debugger",
    "run",
    "legacy",
    "unzip_extra",
    "backup",
    "restore",
    "update",
    "extra",
    "mods",
    "dev",
]

# Files that are excluded from the '--mods' flag.
excluded_files = [
    "Debugger.py",
    "Legal.py",
    "UAC.ps1",
    "UACPY.py",
    "Backup.py",
    "Restore.py",
    "Update.py",
    "Extra_Menu.py",
    "Logicytics.py",
    "Windows_Defender_Crippler.bat",
    "APIGen.py",
    "Structure.py",
    "Crash_Reporter.py",
    "Error_Gen.py",
    "Unzip_Extra.py",
    "Crash_Reporter.py",
    "Error_Gen.py",
]

# Dictionary of flags that conflict each other, and a message.
conflicts = {
    ("mini_log", "silent"): "Both 'mini-log' and 'silent' are used.",
    ("shutdown", "silent"): "Both 'shutdown' and 'silent' are used.",
    ("shutdown", "reboot"): "Both 'shutdown' and 'reboot' are used.",
    ("shutdown", "bios"): "Both 'shutdown' and 'bios' are used.",
    ("reboot", "silent"): "Both 'reboot' and 'silent' are used.",
    ("reboot", "bios"): "Both 'reboot' and 'bios' are used.",
    ("bios", "silent"): "Both 'bios' and 'silent' are used.",
    ("debug", "silent"): "Both 'debug' and 'silent' are used.",
    ("debug", "mini_log"): "Both 'debug' and 'mini-log' are used.",
    ("dev", "silent"): "Both 'dev' and 'silent' are used.",
    ("dev", "mini_log"): "Both 'dev' and 'mini-log' are used.",
}

# Flags that are used to control the files run.
run_actions = {
    "onlypy": "onlypy",
    "setup_only": "setup_only",
    "setup": "setup",
    "minimum": "minimum",
    "only_native": "only_native",
    "debugger_only": "debugger_only",
    "debugger": "debugger",
    "run": "run",
    "legacy": "legacy",
    "unzip_extra": "unzip_extra",
    "backup": "backup",
    "restore": "restore",
    "update": "update",
    "extra": "extra",
    "mods": "mods",
    "dev": "dev",
}

# Flags that are used to control the logging mechanism of the program.
log_actions = {
    "mini_log": "mini_log",
    "silent": "silent",
    "debug": "debug",
}

# Flags that are used to quit the program in a specific way.
quit_actions = {
    "shutdown": "shutdown",
    "reboot": "reboot",
    "bios": "bios",
}

# Admin exception flags that are allowed to ignore the admin check before the program is run.
admin_exceptions = {
    "debugger_only",
    "unzip_extra",
    "backup",
    "restore",
    "update",
    "extra",
}

# Define the supported languages
languages = {
    "py": "Python",
    "js": "JavaScript",
    "bat": "Batch",
    "ps1": "PowerShell",
    "java": "Java",
    "c": "C",
    "cpp": "C++",
    "cs": "C#",
    "go": "Go",
    "rb": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kt": "Kotlin",
    "scala": "Scala",
    "rs": "Rust",
    "ts": "TypeScript",
    "r": "R",
    "lua": "Lua",
    "pl": "Perl",
    "sh": "Shell Script",
    "hs": "Haskell",
    "f90": "Fortran (Fixed)",
    "vbs": "VBScript",
    "asm": "Assembly",
    "ml": "OCaml",
    "fs": "F#",
    "dart": "Dart",
    "groovy": "Groovy",
    "tcl": "Tcl",
    "awk": "AWK",
    "julia": "Julia",
    "erl": "Erlang",
    "nim": "Nim",
    "ada": "Ada",
    "cobol": "COBOL",
    "pascal": "Pascal",
    "prolog": "Prolog",
    "scheme": "Scheme",
    "smalltalk": "Smalltalk",
    "lisp": "Lisp",
    "clojure": "Clojure",
    "crystal": "Crystal",
    "zig": "Zig",
    "d": "D",
    "objc": "Objective-C",
    "as": "ActionScript",
    "cfml": "ColdFusion",
    "apex": "Apex",
    "sol": "Solidity",
    "kotlin": "Kotlin",
    "coffee": "CoffeeScript",
    "pyx": "Cython",
    "jr": "Jython",
    "ipy": "IronPython",
    "booc": "Boo",
    "nem": "Nemerle",
    "valac": "Vala",
    "gn": "Genie",
    "seed7": "Seed7",
    "ob": "Oberon",
    "mod": "Modula-2",
    "pico": "PicoLisp",
    "ar": "Arc",
    "ahk": "AutoHotkey",
    "au3": "AutoIt",
}
