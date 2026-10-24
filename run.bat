@echo off
title SlothML - Tabular Foundation AI
echo ========================================================
echo   Starting SlothML (TabPFN Tabular AI Dashboard)
echo ========================================================
echo.
cd /d "%~dp0"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
