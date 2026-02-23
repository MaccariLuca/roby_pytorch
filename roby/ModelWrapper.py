"""
Model Wrapper Unificato per Keras e PyTorch
Fornisce un'interfaccia comune per lavorare con modelli di entrambi i framework.

@author: Claudio Bors, Claude AI Assistant
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """
    Classe astratta che definisce l'interfaccia comune per wrapper di modelli.
    """
    
    @abstractmethod
    def predict(self, x, verbose=0):
        """Esegue predizione su input x"""
        pass
    
    @abstractmethod
    def get_input_shape(self):
        """Ritorna (height, width, channels)"""
        pass
    
    @abstractmethod
    def get_framework(self):
        """Ritorna 'keras' o 'pytorch'"""
        pass
    
    @property
    def input_shape(self):
        """Property per compatibilità con codice Keras esistente"""
        h, w, c = self.get_input_shape()
        return (None, h, w, c)


class KerasModelWrapper(BaseModelWrapper):
    """Wrapper per modelli Keras/TensorFlow"""
    
    def __init__(self, model):
        self.model = model
        self._framework = 'keras'
        
    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)
    
    def get_input_shape(self):
        try:
            cfg = self.model.input_shape
            if isinstance(cfg, list):
                cfg = cfg[0]
            h = cfg[1]
            w = cfg[2]
            c = cfg[3] if len(cfg) > 3 else 1
            return h, w, c
        except Exception:
            return 224, 224, 3
    
    def get_framework(self):
        return self._framework


class PyTorchModelWrapper(BaseModelWrapper):
    """Wrapper per modelli PyTorch"""
    
    def __init__(self, model, input_shape=None, device='cpu'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self._framework = 'pytorch'
        self._input_shape = input_shape or (224, 224, 3)
    
    def predict(self, x, verbose=0):
        """
        Esegue predizione PyTorch.
        Input: numpy array [B, H, W, C] (formato Keras)
        Output: numpy array [B, num_classes]
        """
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_torch = torch.from_numpy(x).float()
            else:
                x_torch = x.float()
            
            # BHWC -> BCHW
            if x_torch.dim() == 4 and x_torch.shape[-1] in [1, 3, 4]:
                x_torch = x_torch.permute(0, 3, 1, 2)
            
            x_torch = x_torch.contiguous().to(self.device)
            output = self.model(x_torch)
            return output.cpu().numpy()
    
    def get_input_shape(self):
        return self._input_shape
    
    def get_framework(self):
        return self._framework


def wrap_model(model, input_shape=None, device='cpu'):
    """
    Factory function per creare il wrapper appropriato.
    """
    if isinstance(model, BaseModelWrapper):
        return model
    if isinstance(model, nn.Module):
        return PyTorchModelWrapper(model, input_shape=input_shape, device=device)
    else:
        return KerasModelWrapper(model)


def is_wrapped(model):
    """Verifica se un modello è già wrappato"""
    return isinstance(model, BaseModelWrapper)
