@echo off
echo ======================================================================
echo  Uploading KWS Audio Data Engine to GitHub (Repository: kws-audio-data-engine)
echo ======================================================================
echo.

set GIT_CMD=git

echo [1/3] Checking current branch and staging changes...
%GIT_CMD% branch -M main
%GIT_CMD% add .
%GIT_CMD% commit -m "feat: complete KWS synthetic voice generation and dynamic SNR noise augmentation pipelines" >nul 2>&1

echo [2/3] Setting remote repository URL...
%GIT_CMD% remote set-url origin https://github.com/probat1/Voice-data-collector.git

echo [3/3] Pushing to GitHub (main branch)...
%GIT_CMD% push -u origin main --force

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo [SUCCESS] Code and dataset successfully uploaded to GitHub!
    echo Repository: https://github.com/probat1/Voice-data-collector
    echo ======================================================================
) else (
    echo.
    echo [NOTE] If you created a new repository named 'kws-audio-data-engine' on GitHub:
    echo        1. Run: git remote set-url origin https://github.com/probat1/kws-audio-data-engine.git
    echo        2. Run: git push -u origin main --force
)

echo.
pause
