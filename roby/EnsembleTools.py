"""
Ensemble Tools - AGGIORNATO per supporto PyTorch
Modulo per la gestione di Ensemble di modelli.
Supporta caricamento dinamico di Keras (.keras, .h5, .hdf5) e PyTorch (.pt, .pth).

@author: Claudio Bors, garganti, bombarda, Claude AI Assistant
"""

import os
import glob
import torch
import torch.nn as nn

try:
    from tensorflow.keras.models import load_model as keras_load_model, clone_model
except ImportError:
    from keras.models import load_model as keras_load_model, clone_model

from roby.ModelWrapper import wrap_model, is_wrapped, BaseModelWrapper

# Estensioni supportate
KERAS_EXTENSIONS = ['.keras', '.h5', '.hdf5', '.h7', '.pb', '']
PYTORCH_EXTENSIONS = ['.pt', '.pth', '.ckpt']
ALL_EXTENSIONS = KERAS_EXTENSIONS + PYTORCH_EXTENSIONS


def _detect_model_type(path):
    """Rileva il tipo di modello dal path"""
    ext = os.path.splitext(path)[1].lower()
    
    if ext in PYTORCH_EXTENSIONS:
        return 'pytorch'
    elif ext in KERAS_EXTENSIONS or os.path.isdir(path):
        return 'keras'
    return None


def _find_model_file(model_dir, prefix, index=None):
    """Cerca un file modello valido"""
    base_name = f"{prefix}_model"
    if index is not None:
        base_name += f"_{index}"
    
    for ext in ALL_EXTENSIONS:
        if ext == '':
            path = os.path.join(model_dir, base_name)
            if os.path.isdir(path):
                return path
        else:
            path = os.path.join(model_dir, base_name + ext)
            if os.path.isfile(path):
                return path
    
    pattern = os.path.join(model_dir, base_name + ".*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_single_model(path, input_shape=None, device='cpu'):
    """
    Carica un singolo modello (Keras o PyTorch) e lo wrappa.
    
    Args:
        path: percorso file modello
        input_shape: (h, w, c) per PyTorch
        device: 'cpu' o 'cuda'
    
    Returns:
        BaseModelWrapper
    """
    model_type = _detect_model_type(path)
    
    if model_type == 'pytorch':
        model = torch.load(path, map_location=device, weights_only=False)
        
        if isinstance(model, dict) and 'model_state_dict' in model:
            raise ValueError(
                f"Il file {path} contiene solo state_dict. "
                "Fornire l'architettura separatamente."
            )
        
        return wrap_model(model, input_shape=input_shape, device=device)
    
    elif model_type == 'keras':
        model = keras_load_model(path)
        return wrap_model(model)
    
    else:
        raise ValueError(f"Tipo modello non riconosciuto: {path}")


def save_models(models, model_dir, ensemble_models=False, extension=".keras"):
    """Salva lista di modelli"""
    os.makedirs(model_dir, exist_ok=True)
    n = len(models)
    prefix = "ensemble" if ensemble_models else "original"
    
    print(f"Saving {n} models to '{model_dir}'...")
    
    for i, model in enumerate(models):
        name = f"{prefix}_model"
        if n > 1:
            name += f"_{i}"
        
        filename = os.path.join(model_dir, name + extension)
        
        if is_wrapped(model):
            if model.get_framework() == 'pytorch':
                torch.save(model.model, filename)
            else:
                model.model.save(filename)
        else:
            if isinstance(model, nn.Module):
                torch.save(model, filename)
            else:
                model.save(filename)
    
    print("Models saved.")


def load_models(n_models, model_dir, ensemble_models=False, input_shape=None, device='cpu'):
    """
    Carica n modelli da directory e li wrappa.
    
    Args:
        n_models: numero modelli
        model_dir: directory
        ensemble_models: True per "ensemble_model_X"
        input_shape: (h,w,c) per PyTorch
        device: 'cpu' o 'cuda'
    
    Returns:
        List[BaseModelWrapper]
    """
    prefix = "ensemble" if ensemble_models else "original"
    print(f"Loading {n_models} models from '{model_dir}'...")
    
    models = []
    for i in range(n_models):
        path = _find_model_file(model_dir, prefix, index=i)
        
        if path is None and n_models == 1:
            path = _find_model_file(model_dir, prefix, index=None)
        
        if path is None:
            raise FileNotFoundError(
                f"Modello {i} non trovato in '{model_dir}' con prefix '{prefix}'"
            )
        
        print(f"  -> Loading {i+1}/{n_models}: {os.path.basename(path)}")
        model = load_single_model(path, input_shape=input_shape, device=device)
        models.append(model)
    
    print("All models loaded.")
    return models


def train_ensemble_models(base_model, x_train, y_train, compile_config, train_config, n_models):
    """Addestra ensemble (solo Keras)"""
    models = []
    for i in range(n_models):
        print(f"-- Training model {i+1}/{n_models} --")
        model = clone_model(base_model)
        model.compile(**compile_config)
        model.fit(x_train, y_train, **train_config)
        models.append(wrap_model(model))
    return models
