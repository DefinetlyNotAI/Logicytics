# List all optional features and save them to Features.txt
Write-Output "INFO: Starting Feature mining, saving to Features.txt"
Get-WindowsOptionalFeature -Online | Format-Table -Property FeatureName, State > Features.txt
