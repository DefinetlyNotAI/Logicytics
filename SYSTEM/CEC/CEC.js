// This has not been tested!

const fs = require('fs');
const { exec } = require('child_process');

function crash(errorId, functionName, errorContent, type) {
    /**
     * Writes error information to temporary files and executes a crash reporter script.
     *
     * @param {string} errorId - The ID of the error.
     * @param {string} functionName - The name of the function where the error occurred.
     * @param {string} errorContent - The content of the error.
     * @param {string} type - The type of error.
     * @return {void}
     */
    fs.writeFileSync('flag.temp', 'PlaceholderScriptName.js');
    fs.writeFileSync('error.temp', errorId);
    fs.writeFileSync('function.temp', functionName);
    fs.writeFileSync('language.temp', 'JavaScript');
    fs.writeFileSync('error_data.temp', errorContent);
    fs.writeFileSync('type.temp', type);

    exec('powershell.exe -Command "&.\\Crash_Reporter.py"', (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return;
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
    });
}

crash("PE", "fun5", "ERR", "crash");
