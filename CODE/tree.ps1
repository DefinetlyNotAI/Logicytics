Write-Host "INFO: Starting Tree Command"

# Define the output file name as Tree.txt
$outputFile = "Tree.txt"

# Run the tree command and redirect the output to the file
tree /f C:\ | Out-File -FilePath $outputFile -Force

Write-Host "INFO: Saved $outputFile"