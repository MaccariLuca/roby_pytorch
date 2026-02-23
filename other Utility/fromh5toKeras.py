import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Dense, Input, add
from tensorflow.keras.models import Model
import os

# 1. Configurazione percorsi
path_modelli = r'C:\Sviluppo\datasets\Pneumothorax\Models'
nome_file_h5 = 'weights.h5' # Assicurati che sia il file corretto del notebook
input_path = os.path.join(path_modelli, nome_file_h5)
output_path = os.path.join(path_modelli, 'pneumo_model_simple.keras')

print("Inizio ricostruzione modello basato sul Notebook...")

# 2. Ricostruzione esatta dell'architettura del Notebook
inputs = Input(shape=(256, 256, 1))
  
x = Conv2D(32, (5,5), padding='same', activation='relu')(inputs)
y = Conv2D(32, (5,5), padding='same', activation='relu')(x)
z = MaxPooling2D(pool_size=(2,2))(add([x,y]))
  
x = Conv2D(64, (5,5), padding='same', activation='relu')(z)
y = Conv2D(64, (5,5), padding='same', activation='relu')(x)
z = MaxPooling2D(pool_size=(2,2))(add([x,y]))
  
x = SeparableConv2D(128, (5,5), padding='same', activation='relu')(z)
y = SeparableConv2D(128, (5,5), padding='same', activation='relu')(x)
z = MaxPooling2D(pool_size=(2,2))(add([x,y]))
  
x = Conv2D(256, (3,3), padding='same', activation='relu')(z)
y = Conv2D(256, (3,3), padding='same', activation='relu')(x)
y_2 = Conv2D(256, (3,3), padding='same', activation='relu')(y)
z = MaxPooling2D(pool_size=(2,2))(add([x, y_2]))

z = GlobalAveragePooling2D()(z)
x = Dense(64, activation='relu')(z)
predictions = Dense(2, activation='sigmoid')(x) # Output a 2 classi

model = Model(inputs=inputs, outputs=predictions)

# 3. Caricamento dei pesi
try:
    model.load_weights(input_path)
    print("Pesi caricati con successo!")
    model.save(output_path)
    print(f"Modello salvato correttamente in: {output_path}")
except Exception as e:
    print(f"Errore durante il caricamento: {e}")