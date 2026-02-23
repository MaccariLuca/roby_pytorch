@echo off
TITLE Roby Console

cd /d C:\Sviluppo\python311
call Scripts\activate

cd /d C:\Sviluppo\python311\robycode

:: 3. Messaggio di benvenuto
echo ========================================================
echo  comandi possibili:
echo   - python -m roby.run_roby --help
echo   - python -m roby.run_roby robustness ...
echo   - python -m roby.run_roby uncertainty ...
echo   - python -m roby.run_roby referral ...
echo.

:: 4. Lascia il terminale aperto per i tuoi comandi
cmd /k