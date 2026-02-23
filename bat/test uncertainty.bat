@echo off
TITLE Roby - Test INCERTEZZA

set PYTHON_DIR=C:\Sviluppo\python311
set PROJECT_DIR=C:\Sviluppo\python311\robycode
set MODEL=C:\Sviluppo\datasets\mnist_roby\original_model.keras
set IMAGES=C:\Sviluppo\datasets\mnist_roby\test_images
set LABELS=C:\Sviluppo\datasets\mnist_roby\classes.csv
set LOG_FILE=roby_uncertainty.log

set ALT_NAME=Blur
set ALT_PARAMS=0,1,3
set STEPS=11

:: 'entropy'
:: 'mi' o 'mean_entropy' (Lento)
set METHOD=entropy
:: set METHOD=mi
set MC_SAMPLES=10

cd /d %PYTHON_DIR%
call Scripts\activate
set TF_ENABLE_ONEDNN_OPTS=0
cd /d %PROJECT_DIR%

echo ========================================================
echo  AVVIO TEST INCERTEZZA
echo  Metodo: %METHOD%
echo ========================================================

python -m roby.run_roby uncertainty ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --npoints %STEPS% ^
  --unc_method %METHOD% ^
  --n_mc_samples %MC_SAMPLES% ^
  --threshold 0.5 ^
  --logfile "%LOG_FILE%"

echo.
echo Test completato.
pause