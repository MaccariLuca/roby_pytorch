'''
The **Utilitys** module - Optimized for Memory-Safe Multicore Processing
@author: Andrea Bombarda
'''
from sympy import solve   # type: ignore
from sympy.abc import a, b, c   # type: ignore
import numpy as np   # type: ignore
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- PARABOLA UTILS (Usate in RobustnessNN) ---

def compute_parabola(x1: float, y1: float, x2: float, y2: float, xv: float):
    """
    Computes the parameters a,b, and c of a parabola expressed in the shape of
    ax^2 + bx + c = y
    """
    return solve([a*x1**2 + b*x1 + c - y1,
                  a*x2**2 + b*x2 + c - y2,
                  b**2-4*a*c + 4*a*xv], a, b, c)


def compute_appoximate_parabola(x1: float, y1: float, x2: float, y2: float,
                                x3: float, y3: float):
    """
    Computes the parameters a,b, and c of a parabola using polyfit.
    """
    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    parameters = np.polyfit(x, y, 2)
    return parameters


# --- MULTICORE LOADING & PROCESSING UTILS ---

def _load_single_image_task(args):
    """Worker function per caricare una singola immagine."""
    i, f, reader_f, classes, label_list = args
    img = None
    if isinstance(f, str):
        if reader_f: img = reader_f(f)
        else: 
            import cv2
            img = cv2.imread(f)
    else: img = f
    
    label_idx = -1
    if label_list:
        lbl = label_list[i]
        try: label_idx = classes.index(str(lbl))
        except ValueError: label_idx = -1
    return i, img, label_idx

def load_raw_data_multicore(environment, max_workers=None):
    """
    Carica l'intero dataset in parallelo usando ThreadPool.
    Restituisce (x_list, y_arr).
    """
    if not environment.file_list: return [], np.array([])
    
    # Prepara argomenti per i worker
    tasks = [(i, f, environment.reader_f, environment.classes, environment.label_list) 
             for i, f in enumerate(environment.file_list)]
    
    if max_workers is None: max_workers = min(32, (os.cpu_count() or 1) + 4)
    print(f" -> Multicore Loading ({max_workers} threads)...")
    
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for future in as_completed([executor.submit(_load_single_image_task, t) for t in tasks]):
            try:
                idx, img, lbl = future.result()
                results[idx] = (img, lbl)
            except: pass

    # Filtra risultati validi
    valid = [r for r in results if r]
    return [r[0] for r in valid], np.array([r[1] for r in valid])


# --- MULTICORE PROCESSING & PREDICTION ---

def _alter_task(args):
    """Worker function per alterare una singola immagine."""
    img, alt, lvl, pre = args
    if alt and lvl > 0: img = alt.apply_alteration(img, lvl)
    if pre: img = pre(img)
    return img

def process_batch_multicore(x_list, alteration, level, preprocess_f, max_workers=None):
    """
    Applica Alterazione + Preprocessing in parallelo.
    Restituisce un batch numpy.
    """
    if not x_list: return np.array([])
    
    if max_workers is None: max_workers = min(32, (os.cpu_count() or 1) + 4)
    tasks = [(img, alteration, level, preprocess_f) for img in x_list]
    
    processed_imgs = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_alter_task, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futures):
            processed_imgs[futures[f]] = f.result()
            
    return np.vstack(processed_imgs)

def predict_in_chunks(model, x_list, alteration, level, preprocess_f, chunk_size=64):
    """
    Esegue la pipeline (Alter -> Preprocess -> Predict) a blocchi (Chunks).
    Ottimizza la memoria e parallelizza CPU/GPU.
    """
    total = len(x_list)
    all_preds = []
    
    # Auto-tune workers
    max_workers = min(16, (os.cpu_count() or 1) + 2)
    
    # Processa in chunks
    for i in range(0, total, chunk_size):
        chunk_raw = x_list[i : i + chunk_size]
        
        # 1. Prepare Batch (Multithreaded CPU)
        chunk_processed = [None] * len(chunk_raw)
        tasks = [(img, alteration, level, preprocess_f) for img in chunk_raw]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_alter_task, t): k for k, t in enumerate(tasks)}
            for f in as_completed(futures):
                chunk_processed[futures[f]] = f.result()
        
        # Stack numpy array
        if not chunk_processed: continue
        x_batch = np.vstack(chunk_processed)
        
        # 2. Predict (GPU/Vectorized CPU)
        if isinstance(model, list): # Ensemble
            # Shape: (n_models, batch_size, n_classes) -> Transpose to (batch, models, classes)
            batch_preds = np.array([m.predict(x_batch, verbose=0) for m in model])
            batch_preds = np.transpose(batch_preds, (1, 0, 2))
        else: # Single Model
            batch_preds = model.predict(x_batch, verbose=0)
            
        all_preds.append(batch_preds)
        
    return np.concatenate(all_preds, axis=0) if all_preds else np.array([])