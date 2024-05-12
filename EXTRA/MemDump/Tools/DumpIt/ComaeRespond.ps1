Param (
 [Parameter(Mandatory=$true)][string]$Token,
 [Parameter(Mandatory=$true)][string]$OrganizationId,
 [Parameter(Mandatory=$true)][string]$CaseId,
 [Parameter()] [string] $Hostname="beta.comae.tech"
)

$TempDir = [System.IO.Path]::GetTempPath()
Set-Location $TempDir
Write-Host "Current Directory: " $pwd
if (Test-Path -Path Comae-Toolkit.zip) {
    Remove-Item Comae-Toolkit.zip
}

if (Test-Path -Path Comae-Toolkit) {
    Remove-Item Comae-Toolkit\* -Force -Recurse
}

$Headers = @{
    "Authorization" = "Bearer " + $Token;
}
$Uri = "https://" + $Hostname + "/api/download"
Invoke-WebRequest -Uri $Uri -Method GET -OutFile Comae-Toolkit.zip -Headers $Headers

$rootDir = $pwd

if (Test-Path -Path Comae-Toolkit.zip) {
    Expand-Archive -Path Comae-Toolkit.zip -Force
    Set-Location -Path  ".\Comae-Toolkit"
    Import-Module .\Comae.psm1
    $DumpFile = New-ComaeDumpFile -Directory $rootDir\Dumps -IsCompress
    Send-ComaeDumpFile -Token $Token -Path $DumpFile -ItemType File -Hostname $Hostname -OrganizationId $OrganizationId -CaseId $CaseId

    Set-Location $rootDir
    # Clean everything.
    Remove-Item $rootDir\Dumps\* -Force -Recurse
    Remove-Item $rootDir\Comae-Toolkit.zip
    Remove-Item $rootDir\Comae-Toolkit\* -Force -Recurse
}