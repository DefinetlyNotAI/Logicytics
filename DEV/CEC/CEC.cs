/// This has not been tested!

using System.IO;
using System.Diagnostics;

class Program {
    /// <summary>
    /// The entry point of the program.
    /// </summary>
    /// <param name="args">The command line arguments.</param>
    static void Crash(string errorId, string functionName, string errorContent, string type) {
        File.WriteAllText("flag.temp", "PlaceholderScriptName.cs");
        File.WriteAllText("error.temp", errorId);
        File.WriteAllText("function.temp", functionName);
        File.WriteAllText("language.temp", "C#");
        File.WriteAllText("error_data.temp", errorContent);
        File.WriteAllText("type.temp", type);

        var startInfo = new ProcessStartInfo("powershell.exe", "-Command \"&.\\Crash_Reporter.py\"") {
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        var process = new Process { StartInfo = startInfo };
        process.Start();
        while (!process.StandardOutput.EndOfStream) {
            string line = process.StandardOutput.ReadLine();
            Console.WriteLine(line);
        }
    }

    static void Main(string[] args) {
        Crash("PE", "fun5", "ERR", "crash");
    }
}
