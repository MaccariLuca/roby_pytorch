@echo off
TITLE Roby - Test Referral Cost-Sensitive (JSON)

set PYTHON_DIR=C:\Sviluppo\python311
set PROJECT_DIR=C:\Sviluppo\python311\robycode
:: Percorso del file di configurazione JSON
set CONFIG_FILE=C:\Sviluppo\python311\robycode\json config\config_referral_dottore.json

cd /d %PYTHON_DIR%
call Scripts\activate
:: Disabilita log oneDNN per pulizia output
set TF_ENABLE_ONEDNN_OPTS=0
cd /d %PROJECT_DIR%

echo ========================================================
echo  AVVIO TEST REFERRAL (COST SENSITIVE)
echo  Configurazione: "%CONFIG_FILE%"
echo ========================================================

:: Esegue il test leggendo tutti i parametri (inclusi costi e doc_acc) dal JSON
python -m roby.run_roby from_config "%CONFIG_FILE%"

echo.
echo Test completato.
pause