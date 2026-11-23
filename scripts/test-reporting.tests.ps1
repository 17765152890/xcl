# Exercise the real test runner with fake cargo output; no build or GPU is needed.
$ErrorActionPreference = 'Stop'
$runner = (Join-Path $PSScriptRoot 'test.ps1').Replace("'", "''")
$shell = (Get-Process -Id $PID).Path

function Check-Run($Name, $BuildExit, $TestExit, $Markers) {
    $command = @'
function cargo {
    if ($args[0] -eq 'build') {
        $global:LASTEXITCODE = BUILD_EXIT
        return
    }
    'TEST_COMMAND_REACHED'
    if (WITH_MARKERS) {
        'SKIPPED synthetic_gpu: no wave64'
        'SKIPPED_SAMPLE first: XCL_SAMPLE_DLLS not set'
        'SKIPPED_SAMPLE second: XCL_SAMPLE_DLLS not set'
        'SKIPPED_SAMPLE third: XCL_SAMPLE_RPF not set'
    }
    'test ignored_probe ... ignored'
    $global:LASTEXITCODE = TEST_EXIT
}
& 'RUNNER_PATH'
exit $LASTEXITCODE
'@
    $command = $command.Replace('BUILD_EXIT', "$BuildExit").Replace('TEST_EXIT', "$TestExit")
    $command = $command.Replace('WITH_MARKERS', $(if ($Markers) {'$true'} else {'$false'}))
    $command = $command.Replace('RUNNER_PATH', $runner)
    $output = & $shell -NoProfile -Command $command 2>&1
    $code = $LASTEXITCODE
    $text = $output -join "`n"
    $expected = if ($BuildExit) {$BuildExit} else {$TestExit}
    if ($code -ne $expected) {throw "$Name exit $code, expected $expected`n$text"}
    if ($BuildExit) {
        if ($text -match 'TEST_COMMAND_REACHED|GPU tests skipped|Tests skipped for missing samples') {
            throw "$Name ran tests or printed a summary after a build failure`n$text"
        }
    } elseif ($Markers) {
        if (-not $text.Contains('GPU tests skipped on this adapter: 1 (no wave64)')) {
            throw "$Name GPU count missing`n$text"
        }
        $sampleLine = @($output | Where-Object {"$_" -like 'Tests skipped for missing samples*'})
        if ($sampleLine.Count -ne 1 -or
            -not "$($sampleLine[0])".Contains('(included in libtest passed): 3:') -or
            -not "$($sampleLine[0])".Contains('2 (XCL_SAMPLE_DLLS not set)') -or
            -not "$($sampleLine[0])".Contains('1 (XCL_SAMPLE_RPF not set)')) {
            throw "$Name sample count missing or mixed with GPU/ignored tests`n$text"
        }
    } elseif ($text -match 'GPU tests skipped|Tests skipped for missing samples') {
        throw "$Name reported skips without markers`n$text"
    }
    Write-Host "PASS $Name"
}

Check-Run 'separate skip counts' 0 0 $true
Check-Run 'test failure exit preserved' 0 101 $true
Check-Run 'build failure stops tests' 17 0 $true
Check-Run 'no false skip summary' 0 0 $false
