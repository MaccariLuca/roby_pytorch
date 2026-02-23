@echo off
TITLE Roby TEST completo

cd /d C:\Sviluppo\python311
call Scripts\activate

:: Disabilita i log oneDNN per pulizia output
set TF_ENABLE_ONEDNN_OPTS=0

cd /d C:\Sviluppo\python311\robycode

set MODEL=C:\Sviluppo\datasets\mnist_roby\original_model.keras
set IMAGES=C:\Sviluppo\datasets\mnist_roby\test_images
set LABELS=C:\Sviluppo\datasets\mnist_roby\classes.csv
set LOG_FILE=roby_full_suite.log

set ALT_NAME=Blur
set ALT_PARAMS=0,1,3


:: --- TEST 1: ROBUSTEZZA ---
echo [1/3] Esecuzione Test ROBUSTEZZA...
python -m roby.run_roby robustness ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --npoints 11 ^
  --theta 0.8 ^
  --logfile "%LOG_FILE%"

echo.
echo --------------------------------------------------------
echo.

:: --- TEST 2: INCERTEZZA ---
echo [2/3] Esecuzione Test INCERTEZZA (Entropy)...
:: Qui usiamo l'entropia standard su 11 livelli di Blur
python -m roby.run_roby uncertainty ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --npoints 11 ^
  --unc_method entropy ^
  --threshold 0.5 ^
  --logfile "%LOG_FILE%"

echo.
echo --------------------------------------------------------
echo.

:: --- TEST 3: REFERRAL LEARNING ---
echo [3/3] Esecuzione Test REFERRAL...
:: Qui facciamo un test interessante: 
:: Applichiamo un Blur fisso (livello 0.5) e vediamo se l'incertezza
:: (calcolata con MC Dropout / Mutual Info) ci aiuta a scartare gli errori.
python -m roby.run_roby referral ^
  --modelpath "%MODEL%" ^
  --inputpath "%IMAGES%" ^
  --labelfile "%LABELS%" ^
  --altname %ALT_NAME% ^
  --altparams %ALT_PARAMS% ^
  --altlevel 0.5 ^
  --unc_method mi ^
  --n_mc_samples 10 ^
  --steps 20 ^
  --logfile "%LOG_FILE%"

echo.
echo ========================================================
echo  SUITE COMPLETATA!
echo  Tutti i risultati sono stati salvati in: %LOG_FILE%
echo  Controlla i grafici .jpg generati nella cartella corrente.
echo ========================================================
pause