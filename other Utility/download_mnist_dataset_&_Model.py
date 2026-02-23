import os
import cv2
import numpy as np
import csv
from tensorflow.keras.datasets import mnist

# Configurazione percorsi
BASE_DIR = r"C:\Sviluppo\datasets\mnist_roby"
IMG_DIR = os.path.join(BASE_DIR, "test_images")
CSV_FILE = os.path.join(BASE_DIR, "classes.csv")

def save_mnist_for_roby():
    print(f"Scaricamento MNIST e salvataggio in: {BASE_DIR}...")
    
    # 1. Carica Dati
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # 2. Crea Cartelle e CSV
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        
    classes = sorted(list(set(y_test)))
    
    # Scrittura File Classi
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        for c in classes:
            writer.writerow([str(c)])
            # Crea sottocartella per ogni classe (0, 1, 2...)
            class_dir = os.path.join(IMG_DIR, str(c))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    print("Salvataggio immagini in corso...")
    # 3. Salvataggio Immagini
    for i, (img, label) in enumerate(zip(x_test, y_test)):
        # Percorso: datasets/mnist_roby/test_images/7/image_0001.png
        filename = os.path.join(IMG_DIR, str(label), f"image_{i:05d}.png")
        cv2.imwrite(filename, img)
        
        if i % 1000 == 0:
            print(f"Salvate {i} immagini...")

    print(f"\nCOMPLETATO!")
    print(f" -> Immagini: {IMG_DIR}")
    print(f" -> File Classi: {CSV_FILE}")

if __name__ == "__main__":
    save_mnist_for_roby()