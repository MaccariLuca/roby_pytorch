@echo off
TITLE Roby - Test da Configurazione JSON

set PYTHON_DIR=C:\Sviluppo\python311
set PROJECT_DIR=C:\Sviluppo\python311\robycode
:: Percorso del file di configurazione
set CONFIG_FILE=C:\Sviluppo\python311\robycode\json config\config_uncertainty.json
cd /d %PYTHON_DIR%
call Scripts\activate
set TF_ENABLE_ONEDNN_OPTS=0
cd /d %PROJECT_DIR%

echo ========================================================
echo  AVVIO TEST DA CONFIGURAZIONE
echo  File Config: "%CONFIG_FILE%"
echo ========================================================

:: Esegue il comando from_config passando il percorso del JSON
python -m roby.run_roby from_config "%CONFIG_FILE%"

echo.
echo Test completato.
pause