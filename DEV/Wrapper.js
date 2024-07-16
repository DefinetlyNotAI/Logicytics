// Import the 'child_process' module
const { exec } = require('child_process');

// Define the command to execute the Python script
// Assuming Python is added to your PATH environment variable
const command = 'python Downloader.py';

// Execute the command
exec(command, (error, stdout, stderr) => {
    if (error) {
        console.error(`exec error: ${error}`);
        return;
    }
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
});
