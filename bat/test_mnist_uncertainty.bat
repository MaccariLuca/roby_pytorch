@echo off
TITLE Roby MNIST Uncertainty Test

:: 1. Attiva ambiente
cd /d C:\Sviluppo\python311
call Scripts\activate

:: Disabilita i log oneDNN di TensorFlow prima che Python parta
set TF_ENABLE_ONEDNN_OPTS=0

:: 2. Vai alla cartella di Roby
cd /d C:\Sviluppo\python311\robycode

echo Avvio Test Incertezza su MNIST...

:: NOTA: 
:: --modelpath deve puntare al file .keras o .h5 del modello LeNet
:: --inputpath punta alla cartella creata dallo script export_mnist.py
:: --labelfile punta al file csv creato dallo script export_mnist.py

python -m roby.run_roby uncertainty ^
  --modelpath "C:\Sviluppo\datasets\mnist_roby\original_model.keras" ^
  --inputpath "C:\Sviluppo\datasets\mnist_roby\test_images" ^
  --labelfile "C:\Sviluppo\datasets\mnist_roby\classes.csv" ^
  --altname Blur ^
  --altparams 0,1,3 ^
  --npoints 11 ^
  --unc_method entropy ^
  --threshold 0.5 ^
  --logfile roby_mnist_log.log

echo.
echo Test completato. Controlla i grafici .jpg generati nella cartella robycode.
pause