<#
.SYNOPSIS
    LDV Reorientation - Complete Validation Test Suite
    Systematically runs all 4 phases of Tau stability and Stage 3 validation.

.DESCRIPTION
    Phase 1: Diagnose Speech Tau Stability
    Phase 2: Guided Peak Search Mechanism
    Phase 3: Stage 3 Re-validation
    Phase 4: Final Validation Summary

.PARAMETER Phase
    Run specific phase only (1, 2, 3, 4, or "all")

.PARAMETER SkipOnFailure
    Continue to next phase even if current phase fails

.PARAMETER Verbose
    Show detailed output

.EXAMPLE
    .\run_validation_phases.ps1
    # Runs all phases sequentially

.EXAMPLE
    .\run_validation_phases.ps1 -Phase 1
    # Runs only Phase 1

.EXAMPLE
    .\run_validation_phases.ps1 -SkipOnFailure
    # Runs all phases, continues even if one fails
#>

param(
    [ValidateSet("1", "2", "3", "4", "all")]
    [string]$Phase = "all",

    [switch]$SkipOnFailure,

    [switch]$VerboseOutput
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptsDir = Join-Path $ProjectRoot "scripts\validation"
$ResultsDir = Join-Path $ProjectRoot "results"

# Python executable (adjust if needed)
$PythonExe = "python"

# Phase scripts
$PhaseScripts = @{
    "1" = "phase1_tau_stability.py"
    "2" = "phase2_guided_search.py"
    "3" = "phase3_stage3_revalidation.py"
    "4" = "phase4_final_validation.py"
}

$PhaseNames = @{
    "1" = "Phase 1: Tau Stability Diagnosis"
    "2" = "Phase 2: Guided Peak Search"
    "3" = "Phase 3: Stage 3 Re-validation"
    "4" = "Phase 4: Final Validation Summary"
}

$PhaseDecisions = @{
    0 = @{ Status = "PASS"; Color = "Green"; Message = "Success - proceed to next phase" }
    1 = @{ Status = "WARN"; Color = "Yellow"; Message = "Warning - review before proceeding" }
    2 = @{ Status = "FAIL"; Color = "Red"; Message = "Failed - needs investigation" }
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Banner {
    param([string]$Text)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-PhaseBanner {
    param([string]$PhaseNum, [string]$PhaseName)
    $line = "-" * 60
    Write-Host ""
    Write-Host $line -ForegroundColor White
    Write-Host "  [$PhaseNum] $PhaseName" -ForegroundColor White
    Write-Host $line -ForegroundColor White
}

function Write-PhaseResult {
    param(
        [string]$PhaseNum,
        [int]$ExitCode,
        [TimeSpan]$Duration
    )

    $result = $PhaseDecisions[$ExitCode]
    if (-not $result) {
        $result = @{ Status = "UNKNOWN"; Color = "Gray"; Message = "Unknown exit code: $ExitCode" }
    }

    Write-Host ""
    Write-Host "  Phase $PhaseNum Result: " -NoNewline
    Write-Host $result.Status -ForegroundColor $result.Color
    Write-Host "  $($result.Message)" -ForegroundColor $result.Color
    Write-Host "  Duration: $($Duration.ToString('mm\:ss'))" -ForegroundColor Gray
    Write-Host ""
}

function Run-Phase {
    param(
        [string]$PhaseNum,
        [string]$ScriptName
    )

    $scriptPath = Join-Path $ScriptsDir $ScriptName

    if (-not (Test-Path $scriptPath)) {
        Write-Host "  [ERROR] Script not found: $scriptPath" -ForegroundColor Red
        return 2
    }

    Write-PhaseBanner $PhaseNum $PhaseNames[$PhaseNum]
    Write-Host "  Running: $ScriptName" -ForegroundColor Gray
    Write-Host ""

    $startTime = Get-Date

    try {
        if ($VerboseOutput) {
            & $PythonExe $scriptPath
        } else {
            & $PythonExe $scriptPath 2>&1 | ForEach-Object {
                # Filter output - show only important lines
                if ($_ -match "^  (Phase|Decision|DECISION|Status|Pass Rate|Best|Results|Output|ERROR|WARN|SUMMARY)") {
                    Write-Host $_
                } elseif ($_ -match "^\s*Position \d+:") {
                    Write-Host $_
                } elseif ($_ -match "^=+$") {
                    Write-Host $_
                }
            }
        }
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Host "  [ERROR] $($_.Exception.Message)" -ForegroundColor Red
        $exitCode = 2
    }

    $duration = (Get-Date) - $startTime
    Write-PhaseResult $PhaseNum $exitCode $duration

    return $exitCode
}

function Get-LatestResultDir {
    param([string]$PhaseName)

    $phaseDir = Join-Path $ResultsDir $PhaseName
    if (Test-Path $phaseDir) {
        $latestRun = Get-ChildItem $phaseDir -Directory |
                     Where-Object { $_.Name -match "^run_" } |
                     Sort-Object Name -Descending |
                     Select-Object -First 1
        if ($latestRun) {
            return $latestRun.FullName
        }
    }
    return $null
}

function Show-ResultsSummary {
    Write-Banner "RESULTS SUMMARY"

    $phases = @(
        @{ Name = "phase1_tau_stability"; Display = "Phase 1 (Tau Stability)" }
        @{ Name = "phase2_guided_search"; Display = "Phase 2 (Guided Search)" }
        @{ Name = "phase3_stage3_revalidation"; Display = "Phase 3 (Stage 3)" }
        @{ Name = "phase4_final_validation"; Display = "Phase 4 (Final)" }
    )

    Write-Host "  Latest Results:" -ForegroundColor White
    Write-Host ""

    foreach ($phase in $phases) {
        $resultDir = Get-LatestResultDir $phase.Name
        if ($resultDir) {
            Write-Host "  $($phase.Display):" -ForegroundColor Green
            Write-Host "    $resultDir" -ForegroundColor Gray
        } else {
            Write-Host "  $($phase.Display): Not run" -ForegroundColor Yellow
        }
    }

    Write-Host ""
}

# ============================================================================
# Main Execution
# ============================================================================

Write-Banner "LDV Reorientation - Validation Test Suite"

Write-Host "  Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "  Scripts Dir:  $ScriptsDir" -ForegroundColor Gray
Write-Host "  Results Dir:  $ResultsDir" -ForegroundColor Gray
Write-Host "  Python:       $PythonExe" -ForegroundColor Gray
Write-Host ""

# Check Python availability
try {
    $pythonVersion = & $PythonExe --version 2>&1
    Write-Host "  Python Version: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host "  [ERROR] Python not found. Please install Python or update path." -ForegroundColor Red
    exit 1
}

# Check required packages
Write-Host ""
Write-Host "  Checking required packages..." -ForegroundColor Gray

$requiredPackages = @("numpy", "scipy", "matplotlib")
foreach ($pkg in $requiredPackages) {
    try {
        & $PythonExe -c "import $pkg" 2>&1 | Out-Null
        Write-Host "    [OK] $pkg" -ForegroundColor Green
    } catch {
        Write-Host "    [MISSING] $pkg - install with: pip install $pkg" -ForegroundColor Red
        exit 1
    }
}

# Determine phases to run
$phasesToRun = @()
if ($Phase -eq "all") {
    $phasesToRun = @("1", "2", "3", "4")
} else {
    $phasesToRun = @($Phase)
}

Write-Host ""
Write-Host "  Phases to run: $($phasesToRun -join ', ')" -ForegroundColor Cyan
Write-Host ""

# Run phases
$results = @{}
$startTime = Get-Date

foreach ($p in $phasesToRun) {
    $scriptName = $PhaseScripts[$p]
    $exitCode = Run-Phase $p $scriptName
    $results[$p] = $exitCode

    # Check if we should stop
    if ($exitCode -eq 2 -and -not $SkipOnFailure) {
        Write-Host "  [STOP] Phase $p failed. Use -SkipOnFailure to continue anyway." -ForegroundColor Red
        break
    }

    # Decision point pause (except for last phase)
    if ($p -ne $phasesToRun[-1] -and $exitCode -eq 1) {
        Write-Host "  [NOTE] Phase $p completed with warnings." -ForegroundColor Yellow
        Write-Host "         Review results before proceeding." -ForegroundColor Yellow

        if (-not $SkipOnFailure) {
            $continue = Read-Host "  Continue to next phase? (Y/n)"
            if ($continue -eq "n" -or $continue -eq "N") {
                Write-Host "  [STOP] User requested stop." -ForegroundColor Yellow
                break
            }
        }
    }
}

$totalDuration = (Get-Date) - $startTime

# Final summary
Write-Banner "EXECUTION COMPLETE"

Write-Host "  Total Duration: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host ""
Write-Host "  Phase Results:" -ForegroundColor White

foreach ($p in $phasesToRun) {
    if ($results.ContainsKey($p)) {
        $result = $PhaseDecisions[$results[$p]]
        if (-not $result) {
            $result = @{ Status = "UNKNOWN"; Color = "Gray" }
        }
        Write-Host "    Phase $p : " -NoNewline
        Write-Host $result.Status -ForegroundColor $result.Color
    }
}

Write-Host ""

# Show results locations
Show-ResultsSummary

# Final status
$finalPhase = $phasesToRun[-1]
if ($results.ContainsKey($finalPhase)) {
    $finalCode = $results[$finalPhase]

    if ($finalCode -eq 0) {
        Write-Host "  FINAL STATUS: SUCCESS" -ForegroundColor Green
    } elseif ($finalCode -eq 1) {
        Write-Host "  FINAL STATUS: PARTIAL SUCCESS (review warnings)" -ForegroundColor Yellow
    } else {
        Write-Host "  FINAL STATUS: NEEDS WORK" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan

exit $results[$finalPhase]
