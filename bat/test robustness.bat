@echo off
TITLE Roby - Test ROBUSTEZZA

set PYTHON_DIR=C:\Sviluppo\python311
set PROJECT_DIR=C:\Sviluppo\python311\robycode
set MODEL=C:\Sviluppo\datasets\mnist_roby\original_model.keras
set IMAGES=C:\Sviluppo\datasets\mnist_roby\test_images
set LABELS=C:\Sviluppo\datasets\mnist_roby\classes.csv
set LOG_FILE=roby_robustness.log


set ALT_NAME=Blur
set ALT_PARAMS=0,1,3
set STEPS=11
set THETA=0.8

cd /d %PYTHON_DIR%
call Scripts\activate
set TF_ENABLE_ONEDNN_OPTS=0
cd /d %PROJECT_DIR%

echo ========================================================
echo  AVVIO TEST ROBUSTEZZA
echo  Alterazione: %ALT_NAME% (%ALT_PARAMS%)
echo ========================================================

python -m roby.run_roby robustness ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --npoints %STEPS% ^
  --theta %THETA% ^
  --logfile "%LOG_FILE%"

echo.
echo Test completato.
pause