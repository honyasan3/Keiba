@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Users\tasah\AppData\Local\Programs\Python\Python311\python.exe" predict.py %1 >> "logs\scheduled\predict_%1.log" 2>&1
