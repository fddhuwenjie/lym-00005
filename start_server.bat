@echo off
cd /d "e:\solo\项目\lym-00005"
echo Starting server...
"C:\Users\Huwenjie\AppData\Local\Microsoft\WindowsApps\python.exe" main.py > server_log.txt 2>&1
echo Server exited with code %errorlevel%
type server_log.txt
pause
