# Get all network details
Get-NetAdapter | Select-Object Name, Status, MacAddress, ifIndex, InterfaceAlias, InterfaceDescription | Out-File -FilePath .\Network.txt
