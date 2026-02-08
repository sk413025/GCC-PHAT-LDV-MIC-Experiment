# run_chirp_ref_validation.ps1
# =============================
# Orchestration script for Chirp Reference Validation experiment.
# Runs the 5-step pipeline defined in EXPERIMENT_DESIGN.md.
#
# Usage:
#   .\run_chirp_ref_validation.ps1               # Run all steps
#   .\run_chirp_ref_validation.ps1 -Step 0       # Run only Step 0
#   .\run_chirp_ref_validation.ps1 -Step 2 -SkipOnFailure  # Skip failing steps

param(
    [int]$Step = -1,            # -1 = run all steps
    [switch]$SkipOnFailure,     # Continue even if a step fails
    [switch]$VerboseOutput      # Show full Python output
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"

# Timestamp for this run
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultsBase = Join-Path $ScriptDir "results"

function Write-StepHeader {
    param([int]$StepNum, [string]$Title)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Step $StepNum : $Title" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
}

function Run-Step {
    param(
        [int]$StepNum,
        [string]$Title,
        [string]$Command
    )

    if ($Step -ge 0 -and $Step -ne $StepNum) {
        Write-Host "  [SKIP] Step $StepNum : $Title (not selected)" -ForegroundColor DarkGray
        return $true
    }

    Write-StepHeader $StepNum $Title

    try {
        if ($VerboseOutput) {
            Invoke-Expression $Command
        } else {
            $output = Invoke-Expression $Command 2>&1
            # Show last 20 lines of output
            $lines = $output -split "`n"
            if ($lines.Count -gt 20) {
                Write-Host "  ... ($($lines.Count - 20) lines omitted, use -VerboseOutput to see all)"
                $lines[-20..-1] | ForEach-Object { Write-Host "  $_" }
            } else {
                $lines | ForEach-Object { Write-Host "  $_" }
            }
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Step $StepNum exited with code $LASTEXITCODE"
        }

        Write-Host ""
        Write-Host "  [PASS] Step $StepNum completed successfully." -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host ""
        Write-Host "  [FAIL] Step $StepNum failed: $_" -ForegroundColor Red
        if (-not $SkipOnFailure) {
            Write-Host "  Pipeline aborted. Use -SkipOnFailure to continue." -ForegroundColor Yellow
            exit 1
        }
        return $false
    }
}

# ============================================================================
# Pipeline
# ============================================================================

Write-Host ""
Write-Host "Chirp Reference Validation Pipeline" -ForegroundColor White
Write-Host "Run timestamp: $Timestamp"
Write-Host "Results base:  $ResultsBase"
Write-Host ""

$results = @{}

# --- Step 0: Chirp Cross-Validation (prerequisite) ---
$outDir0 = Join-Path $ResultsBase "chirp_cross_validation"
$results[0] = Run-Step 0 "Chirp Cross-Validation" `
    "$Python `"$ScriptDir\scripts\validation\chirp_reference.py`" --out_dir `"$outDir0`""

# --- Step 1: Phase 1 Re-run with Chirp Reference ---
$outDir1 = Join-Path $ResultsBase "chirp_ref_phase1\run_$Timestamp"
$results[1] = Run-Step 1 "Phase 1: Tau Stability with Chirp Reference" `
    "$Python `"$ScriptDir\scripts\validation\phase1_chirp_reference.py`" --out_dir `"$outDir1`""

# --- Step 2: Phase 2-4 Re-run ---
$outDir2 = Join-Path $ResultsBase "chirp_ref_phase2\run_$Timestamp"
$outDir3 = Join-Path $ResultsBase "chirp_ref_phase3\run_$Timestamp"
$outDir4 = Join-Path $ResultsBase "chirp_ref_phase4\run_$Timestamp"
$results[2] = Run-Step 2 "Phase 2-4: Guided Search, Stage 3, Final Summary" `
    "$Python `"$ScriptDir\scripts\validation\phase2_guided_search.py`" --out_dir `"$outDir2`" ; $Python `"$ScriptDir\scripts\validation\phase3_stage3_revalidation.py`" --out_dir `"$outDir3`" ; $Python `"$ScriptDir\scripts\validation\phase4_final_validation.py`" --out_dir `"$outDir4`""

# --- Step 3: LDV Delay Re-evaluation ---
$outDir5 = Join-Path $ResultsBase "ldv_delay_reeval\run_$Timestamp"
$results[3] = Run-Step 3 "LDV Delay Re-evaluation" `
    "$Python `"$ScriptDir\scripts\validation\ldv_delay_reeval.py`" --out_dir `"$outDir5`""

# --- Step 4: Negative Position Diagnosis ---
$outDir6 = Join-Path $ResultsBase "negative_pos_diagnosis\run_$Timestamp"
$results[4] = Run-Step 4 "Negative Position Diagnosis (-0.4, -0.8)" `
    "$Python `"$ScriptDir\scripts\validation\negative_position_diagnosis.py`" --out_dir `"$outDir6`""

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor White
Write-Host "  Pipeline Summary" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor White

$stepNames = @(
    "Step 0: Chirp Cross-Validation",
    "Step 1: Phase 1 with Chirp Reference",
    "Step 2: Phase 2-4 Re-run",
    "Step 3: LDV Delay Re-evaluation",
    "Step 4: Negative Position Diagnosis"
)

for ($i = 0; $i -lt $stepNames.Count; $i++) {
    $status = if ($results[$i] -eq $true) { "[PASS]" } elseif ($results[$i] -eq $false) { "[FAIL]" } else { "[SKIP]" }
    $color = if ($results[$i] -eq $true) { "Green" } elseif ($results[$i] -eq $false) { "Red" } else { "DarkGray" }
    Write-Host "  $status $($stepNames[$i])" -ForegroundColor $color
}

$passed = ($results.Values | Where-Object { $_ -eq $true }).Count
$total = ($results.Values | Where-Object { $_ -ne $null }).Count
Write-Host ""
Write-Host "  $passed / $total steps passed." -ForegroundColor White
Write-Host "  Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
