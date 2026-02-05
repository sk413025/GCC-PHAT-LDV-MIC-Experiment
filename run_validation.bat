@echo off
REM LDV Reorientation - Validation Test Suite
REM Simple batch file wrapper for run_validation_phases.ps1
REM
REM Usage:
REM   run_validation.bat           - Run all phases
REM   run_validation.bat 1         - Run Phase 1 only
REM   run_validation.bat 2         - Run Phase 2 only
REM   run_validation.bat 3         - Run Phase 3 only
REM   run_validation.bat 4         - Run Phase 4 only
REM   run_validation.bat all       - Run all phases
REM   run_validation.bat all skip  - Run all phases, skip on failure

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo ======================================================================
echo   LDV Reorientation - Validation Test Suite
echo ======================================================================
echo.

REM Parse arguments
set PHASE=all
set SKIP_FLAG=

if not "%1"=="" (
    set PHASE=%1
)

if /i "%2"=="skip" (
    set SKIP_FLAG=-SkipOnFailure
)

REM Run PowerShell script
echo Running: run_validation_phases.ps1 -Phase %PHASE% %SKIP_FLAG%
echo.

powershell -ExecutionPolicy Bypass -File "run_validation_phases.ps1" -Phase %PHASE% %SKIP_FLAG%

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ======================================================================
if %EXIT_CODE%==0 (
    echo   VALIDATION COMPLETE - SUCCESS
) else if %EXIT_CODE%==1 (
    echo   VALIDATION COMPLETE - PARTIAL SUCCESS
) else (
    echo   VALIDATION COMPLETE - NEEDS REVIEW
)
echo ======================================================================
echo.

pause
exit /b %EXIT_CODE%
