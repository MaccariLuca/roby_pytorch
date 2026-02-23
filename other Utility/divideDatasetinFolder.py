import pandas as pd
import os
import shutil

# Configurazione percorsi basata sulla tua cartella
base_path = r'C:\Sviluppo\datasets\Pneumothorax\small_train_data_set'
csv_path = os.path.join(base_path, 'train_data.csv')

# Cartelle di destinazione (verranno create dentro quella principale)
dest_positivo = os.path.join(base_path, 'Positivo')
dest_negativo = os.path.join(base_path, 'Negativo')

# Crea le cartelle se non esistono
os.makedirs(dest_positivo, exist_ok=True)
os.makedirs(dest_negativo, exist_ok=True)

# Carica il dataset
df = pd.read_csv(csv_path)

print("Inizio smistamento delle immagini...")
count_pos = 0
count_neg = 0

for index, row in df.iterrows():
    # Estrae nome file e target dal tuo CSV 
    filename = row['file_name']
    target = row['target']
    
    src_file = os.path.join(base_path, filename)
    
    # Verifica che l'immagine esista nella cartella prima di spostarla
    if os.path.exists(src_file):
        if target == 1:
            shutil.move(src_file, os.path.join(dest_positivo, filename))
            count_pos += 1
        else:
            shutil.move(src_file, os.path.join(dest_negativo, filename))
            count_neg += 1
    else:
        # Utile nel caso il CSV elenchi file non presenti nella cartella small
        continue

print(f"Operazione completata con successo!")
print(f"Immagini spostate in 'Positivo' (Pneumotorace): {count_pos}")
print(f"Immagini spostate in 'Negativo' (Normali): {count_neg}")