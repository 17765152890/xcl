# The name an xcl-smoke block prints. The folder leaf is lowercased and hashed (SHA-256,
# UTF-8, first 8 hex chars) so a PR description does not publish it. A -Title label is printed
# as given.
function Get-XclSmokeTitle([string]$Game, [string]$Title) {
    if (-not [string]::IsNullOrWhiteSpace($Title)) { return $Title }
    $leaf = [string](Split-Path -Path $Game -Leaf)
    if ([string]::IsNullOrEmpty($leaf)) { $leaf = $Game }
    $leaf = $leaf.ToLowerInvariant()
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hex = [System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($leaf)))
    } finally {
        $sha.Dispose()
    }
    return $hex.Replace('-', '').ToLowerInvariant().Substring(0, 8)
}
