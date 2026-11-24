# Run with powershell.exe. Checks the title hash without starting xcl or a game.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'smoke_title.ps1')

function Assert-Eq($name, $got, $want) {
    if ($got -ne $want) { throw "$name : got '$got' wanted '$want'" }
}

# SHA-256 (UTF-8) of the lowercased leaf, first 8 hex chars. Independent of the script's
# implementation: these are the hashes of those strings.
Assert-Eq 'leaf' (Get-XclSmokeTitle -Game 'C:\Games\xcl-smoke-game') 'c809e97b'
Assert-Eq 'case' (Get-XclSmokeTitle -Game 'D:\other\XCL-SMOKE-GAME') 'c809e97b'
Assert-Eq 'spaced' (Get-XclSmokeTitle -Game 'C:\copies\Retail Title') '0fbb8bac'
Assert-Eq 'utf8' (Get-XclSmokeTitle -Game "C:\copies\caf$([char]0x00E9)") '850f7dc4'
Assert-Eq 'label' (Get-XclSmokeTitle -Game 'C:\Games\SecretName' -Title 'retail-title') 'retail-title'
Assert-Eq 'blank label' (Get-XclSmokeTitle -Game 'C:\Games\one' -Title '   ') '7692c3ad'

$one = Get-XclSmokeTitle -Game 'C:\Games\one'
$two = Get-XclSmokeTitle -Game 'C:\Games\two'
if ($one -eq $two) { throw 'different folder names hashed the same' }
if ($one -notmatch '^[0-9a-f]{8}$' -or $two -notmatch '^[0-9a-f]{8}$') { throw "not 8 hex chars: $one $two" }

Write-Output 'smoke title tests passed'
