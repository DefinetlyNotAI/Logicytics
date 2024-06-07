# Understanding the SYSTEM Directory

The `SYSTEM` directory within this project is designed to house the resources that are used to check the integrity of the final product in bug reports, the primary functionalities provided by the main software. These tools are applications designed to test the system, please don't modify any parts of it.


## Understanding the files

### structure.py for creating .structure files

**Tool Description:** `structure.py` is a utility designed for creating the structure files. It's a tool that can be used for debugging purposes, this is not a public tool, but could become public if needed!

### Logicystics.structure for debugging files

**Tool Description:** `Logicystics.structure` is a data file designed for checking the structure files. It's not a tool, but is a requirement for the debugger so that it can be used for debugging purposes.

### Logicystics.version for debugging files

**Tool Description:** `Logicystics.version` is a data file designed for saving the version of the software. It's not a tool, but is a requirement for the debugger so that it can be used for debugging purposes.

### API.KEY for the IP API

**Tool Description:** `API.KEY` is a data file designed for saving the api key of the software. It's not a tool, but is a requirement for the API IP Scraper.

### DEV.pass for developers

**Tool Description:** `DEV.pass` is a data file designed for skipping the ToS checks and API checks. It's not a tool and not included, you can create an empty file to gain DEV permissions, do note that it will produce errors for API keys if they don't exist, I even don't use them!.


## Important Notes

**Testing:** Do not use, modify or run any files in this directory, they don't affect the main functionality of the code, but are necessary for it to function.

## Conclusion

The `SYSTEM` directory is a resource for creating updates and looking to explore additional functionalities and tools that complement the main software can provide destructive. By following the provided instructions and never using these tools, users can ensure proper system management.
