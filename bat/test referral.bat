@echo off
TITLE Roby - Test REFERRAL LEARNING

set PYTHON_DIR=C:\Sviluppo\python311
set PROJECT_DIR=C:\Sviluppo\python311\robycode
set MODEL=C:\Sviluppo\datasets\mnist_roby\original_model.keras
set IMAGES=C:\Sviluppo\datasets\mnist_roby\test_images
set LABELS=C:\Sviluppo\datasets\mnist_roby\classes.csv
set LOG_FILE=roby_referral.log

set ALT_PARAMS=0,1,3
set ALT_LEVEL=0.5

:: Monte Carlo Dropout
set MC_SAMPLES=10
set CURVE_STEPS=20

cd /d %PYTHON_DIR%
call Scripts\activate
set TF_ENABLE_ONEDNN_OPTS=0
cd /d %PROJECT_DIR%

echo ========================================================
echo  AVVIO TEST REFERRAL
echo  Alterazione Fissa: %ALT_NAME% (Livello %ALT_LEVEL%)
echo  Metodo Incertezza: %METHOD%
echo ========================================================

python -m roby.run_roby referral ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --altlevel %ALT_LEVEL% ^
  --unc_method %METHOD% ^
  --n_mc_samples %MC_SAMPLES% ^
  --steps %CURVE_STEPS% ^
  --logfile "%LOG_FILE%"

echo.
echo Test completato.
pause