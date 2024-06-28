#include <stdio.h>
#include <stdlib.h>

void crash(const char* errorId, const char* functionName, const char* errorContent, const char* type) {
    FILE* flagFile;
    FILE* errorFile;
    FILE* functionFile;
    FILE* languageFile;
    FILE* errorDataFile;
    FILE* typeFile;

    // Attempt to open each file using fopen_s
    if (fopen_s(&flagFile, "flag.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open flag.temp file\n");
        return;
    }

    if (fopen_s(&errorFile, "error.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open error.temp file\n");
        return;
    }

    if (fopen_s(&functionFile, "function.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open function.temp file\n");
        return;
    }

    if (fopen_s(&languageFile, "language.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open language.temp file\n");
        return;
    }

    if (fopen_s(&errorDataFile, "error_data.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open error_data.temp file\n");
        return;
    }

    if (fopen_s(&typeFile, "type.temp", "w")!= 0) {
        fprintf(stderr, "Failed to open type.temp file\n");
        return;
    }

    // Write the data to the files
    if (fprintf(flagFile, "%s", "PlaceholderScriptName.c") < 0) {
        perror("Error writing to flag.temp");
    }
    fclose(flagFile);

    if (fprintf(errorFile, "%s", errorId) < 0) {
        perror("Error writing to error.temp");
    }
    fclose(errorFile);

    if (fprintf(functionFile, "%s", functionName) < 0) {
        perror("Error writing to function.temp");
    }
    fclose(functionFile);

    if (fprintf(languageFile, "%s", type) < 0) {
        perror("Error writing to language.temp");
    }
    fclose(languageFile);

    if (fprintf(errorDataFile, "%s", errorContent) < 0) {
        perror("Error writing to error_data.temp");
    }
    fclose(errorDataFile);

    if (fprintf(typeFile, "%s", type) < 0) {
        perror("Error writing to type.temp");
    }
    fclose(typeFile);

    // Execute the crash reporter
    system("powershell.exe -Command \"&.\\Crash_Reporter.py\"");
}

int main() {
    crash("PE", "fun5", "ERR", "crash");
    return 0;
}
