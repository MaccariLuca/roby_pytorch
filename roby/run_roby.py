'''
Roby Unified Runner
Updated to support JSON Configuration, HTML Reporting, Stratified Dataset Slicing, and PyTorch models

Features:
- Robustness Analysis (Single Model)
- Uncertainty Analysis (Ensemble / MCD)
- Referral Learning (Rejection Curves)
- Adaptive Preprocessing
- JSON Configuration Support
- Automatic HTML Reporting
- **Stratified Dataset Partialization** (Balanced sampling per class)
- **PyTorch Model Support** (NEW!)

@author: garganti, bombarda, Claudio Bors
'''
import sys
import os
import logging
import csv
import click
import numpy as np
import cv2
import importlib
import json
import random
import math
from datetime import datetime
from collections import defaultdict

# Importazione modulo Reporter
try:
    from roby.Reporter import HTMLReporter
except ImportError:
    HTMLReporter = None
    print("WARNING: roby.Reporter not found. HTML reports will be disabled.")

# --- CONFIGURAZIONE SILENZIOSA ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_USE_LEGACY_KERAS"] = "1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- FIX PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- IMPORT MODELLI (Keras e PyTorch) ---
try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    from keras.models import load_model as keras_load_model

# Import per supporto PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. PyTorch models will not be supported.")

from imutils import paths
from roby import EnvironmentRTest, Alterations, EnsembleTools
from roby import RobustnessNN, UncertaintyNN, ReferralNN
from roby.viz import referral as viz_ref

# Import wrapper per modelli
try:
    from roby.ModelWrapper import wrap_model, is_wrapped
    from roby.EnsembleTools import load_single_model
    WRAPPER_AVAILABLE = True
except ImportError:
    WRAPPER_AVAILABLE = False
    print("WARNING: ModelWrapper not found. PyTorch support limited.")

# --- LOGGING SETUP ---
def setup_logger(log_file="roby_execution.log"):
    logger = logging.getLogger("RobyLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def log_summary(logger, results, mode="Robustness"):
    logger.info("-" * 40)
    logger.info(f"--- {mode.upper()} EXECUTION SUMMARY ---")
    logger.info("-" * 40)
    if hasattr(results, 'accuracies') and results.accuracies:
        logger.info(f"Average Accuracy: {np.mean(results.accuracies):.4f}")
    if mode == "Referral" and hasattr(results, 'ause'):
        logger.info(f"AUSE Score: {results.ause:.5f}")
    logger.info("=" * 40 + "\n")


# --- FUNZIONI DI SUPPORTO ---

def load_classes_robust(file_path):
    classes = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File classi non trovato: {file_path}") 
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and len(row) > 0 and row[0].strip():
                classes.append(row[0].strip())
    if not classes: raise ValueError("File classi vuoto.")
    return classes

def reader(file_name):
    return cv2.imread(file_name)

def labeler(image_path):
    base_name = os.path.basename(image_path)
    label_from_name = (base_name.split('.')[0]).split('_')[-1]
    parent_folder = os.path.basename(os.path.dirname(image_path))
    if label_from_name.isdigit() or len(label_from_name) < 3: 
        return parent_folder
    if parent_folder in base_name:
        return parent_folder
    return label_from_name

# --- ADAPTIVE PREPROCESSING FACTORY ---

def get_model_input_shape(model):
    '''
    Estrae input shape da modello (supporta wrapper e modelli raw).
    Returns: (height, width, channels)
    '''
    # Se è un wrapper, usa il metodo del wrapper
    if WRAPPER_AVAILABLE and is_wrapped(model):
        return model.get_input_shape()
    
    # Fallback per Keras raw
    try:
        cfg = model.input_shape
        if isinstance(cfg, list): cfg = cfg[0]
        h = cfg[1]
        w = cfg[2]
        c = cfg[3] if len(cfg) > 3 else 1
        return h, w, c
    except Exception as e:
        print(f"WARNING: Impossibile rilevare shape modello ({e}). Uso default 224x224 RGB.")
        return 224, 224, 3

def create_preprocessing_f(target_h, target_w, target_c):
    def dynamic_preprocess(image):
        image = cv2.resize(image, (target_w, target_h))
        if target_c == 1:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
        elif target_c == 3:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    return dynamic_preprocess

def load_alteration(module_name, altname, altparams):
    if module_name is None: module_name = "roby.Alterations"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        try: import roby.Alterations as module
        except ImportError: raise ImportError(f"Impossibile trovare {module_name}")

    if not hasattr(module, altname): raise AttributeError(f"Alterazione '{altname}' non trovata.")
    alterType = getattr(module, altname)
    
    if isinstance(altparams, str):
        params = [p.strip() for p in altparams.split(",")]
        parsed_params = []
        if params != ['']:
            for p in params:
                if '.' in p: parsed_params.append(float(p))
                else: parsed_params.append(int(p))
    elif isinstance(altparams, list):
        parsed_params = altparams
    else:
        parsed_params = []

    try: return alterType(*parsed_params)
    except TypeError: return alterType(parsed_params)


def _load_model_unified(modelpath, input_shape=None, device='cpu'):
    """
    Carica un modello (Keras o PyTorch) e lo wrappa.
    
    Args:
        modelpath: percorso al modello
        input_shape: (h, w, c) per PyTorch (opzionale)
        device: 'cpu' o 'cuda' per PyTorch
    
    Returns:
        Model wrappato o raw (se wrapper non disponibile)
    """
    # Determina il tipo dal path
    is_pytorch = modelpath.endswith('.pt') or modelpath.endswith('.pth')
    
    if is_pytorch:
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch non installato. Installa con: pip install torch")
        
        if WRAPPER_AVAILABLE:
            # Usa il wrapper
            model = load_single_model(modelpath, input_shape=input_shape, device=device)
        else:
            # Fallback: carica raw
            model = torch.load(modelpath, map_location=device, weights_only=False)
            print("WARNING: ModelWrapper non disponibile. Usando modello PyTorch raw.")
    else:
        # Modello Keras
        model = keras_load_model(modelpath)
        
        if WRAPPER_AVAILABLE:
            model = wrap_model(model)
    
    return model


# --- CORE LOGIC EXECUTION ---

def _prepare_file_list(inputpath, max_samples, logger):
    """
    Carica i file. Se max_samples > 0, esegue un campionamento stratificato:
    divide il budget (max_samples) per il numero di classi e preleva 
    lo stesso numero di immagini da ogni cartella.
    """
    all_files = sorted(list(paths.list_images(inputpath)))
    if not all_files: 
        logger.error("Nessuna immagine trovata.")
        return []

    if max_samples > 0:
        # 1. Raggruppa per classe (cartella padre)
        class_files = defaultdict(list)
        for f in all_files:
            # Assumiamo struttura: dataset/classe/immagine.png
            class_name = os.path.basename(os.path.dirname(f))
            class_files[class_name].append(f)
        
        classes = sorted(list(class_files.keys()))
        n_classes = len(classes)
        
        if n_classes > 0:
            # 2. Calcola campioni per classe (arrotondato per eccesso)
            samples_per_class = math.ceil(max_samples / n_classes)
            logger.info(f"Stratified Sampling: {n_classes} classi trovate. Prelevo {samples_per_class} immagini per classe.")
            
            selected_files = []
            random.seed(42) # Seed fisso per riproducibilità tra le classi
            
            for cls in classes:
                files = class_files[cls]
                # Mescola i file di questa classe
                random.shuffle(files)
                # Prendi i primi N
                selected_files.extend(files[:samples_per_class])
            
            # 3. Mescola il risultato finale per evitare ordine sequenziale (0,0,0... 1,1,1...)
            random.shuffle(selected_files)
            
            file_list = selected_files
            print(f" > Dataset parzializzato (Stratified): {len(file_list)} immagini (Target richiesto: {max_samples}).")
        else:
            # Fallback se non ci sono sottocartelle (dataset piatto)
            random.seed(42)
            random.shuffle(all_files)
            file_list = all_files[:max_samples]
    else:
        file_list = all_files

    return file_list

def execute_robustness(params):
    logfile = params.get('logfile', 'roby_robustness.log')
    logger = setup_logger(logfile)
    altname = params['altname']
    
    logger.info(f"STARTING ROBUSTNESS: {altname}")
    print(f"\n--- ROBUSTEZZA: {altname} ---")
    
    try:
        alteration = load_alteration(params.get('module_name', "roby.Alterations"), altname, params['altparams'])
        classes = load_classes_robust(params['labelfile'])
        
        # Carica modello (Keras o PyTorch)
        modelpath = params['modelpath']
        input_shape_hint = params.get('input_shape', None)
        device = params.get('device', 'cpu')
        
        model = _load_model_unified(modelpath, input_shape=input_shape_hint, device=device)
        
        h, w, c = get_model_input_shape(model)
        logger.info(f"Model detected shape: ({h}x{w}), Channels: {c}")
        preprocess_f = create_preprocessing_f(h, w, c)
        
        file_list = _prepare_file_list(params['inputpath'], int(params.get('max_samples', 0)), logger)
        if not file_list: return

        env = EnvironmentRTest.EnvironmentRTest(
            model=model,
            file_list=file_list,
            classes=classes,
            labeler_f=labeler,
            reader_f=reader,
            preprocess_f=preprocess_f
        )
        
        results = RobustnessNN.robustness_test(env, alteration, int(params['npoints']), float(params['theta']))
        log_summary(logger, results, mode="Robustness")
        RobustnessNN.display_robustness_results(results)

        if params.get('report', True) and HTMLReporter:
            report_file = logfile.replace('.log', '.html')
            reporter = HTMLReporter(filename=report_file)
            reporter.generate_report(results, mode="Robustness")
    
    except Exception as e:
        logger.critical(f"Execution Error: {e}")
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()

def execute_uncertainty(params):
    logfile = params.get('logfile', 'roby_uncertainty.log')
    logger = setup_logger(logfile)
    altname = params['altname']
    
    logger.info(f"STARTING UNCERTAINTY: {altname}")
    print(f"\n--- INCERTEZZA: {altname} ---")

    try:
        alteration = load_alteration(params.get('module_name', "roby.Alterations"), altname, params['altparams'])
        classes = load_classes_robust(params['labelfile'])
        
        file_list = _prepare_file_list(params['inputpath'], int(params.get('max_samples', 0)), logger)
        if not file_list: return
        
        env = None
        models_dir = params.get('models_dir')
        n_models = int(params.get('n_models', 0))
        input_shape_hint = params.get('input_shape', None)
        device = params.get('device', 'cpu')
        
        if models_dir and n_models > 0:
            # Carica ensemble
            ensemble = EnsembleTools.load_models(n_models, models_dir, ensemble_models=True,
                                                 input_shape=input_shape_hint, device=device)
            h, w, c = get_model_input_shape(ensemble[0])
            preprocess_f = create_preprocessing_f(h, w, c)
            env = EnvironmentRTest.EnvironmentRTest(ensemble_models=ensemble, file_list=file_list, 
                                                     classes=classes, labeler_f=labeler, 
                                                     reader_f=reader, preprocess_f=preprocess_f)
        else:
            # Carica modello singolo
            modelpath = params['modelpath']
            model = _load_model_unified(modelpath, input_shape=input_shape_hint, device=device)
            
            h, w, c = get_model_input_shape(model)
            preprocess_f = create_preprocessing_f(h, w, c)
            env = EnvironmentRTest.EnvironmentRTest(model=model, file_list=file_list, 
                                                     classes=classes, labeler_f=labeler, 
                                                     reader_f=reader, preprocess_f=preprocess_f)
        
        results = UncertaintyNN.uncertainty_test(
            env, alteration, 
            int(params['npoints']), 
            params.get('unc_method', 'entropy'), 
            float(params.get('threshold', 0.5)), 
            int(params.get('n_mc_samples', 10))
        )
        log_summary(logger, results, mode="Uncertainty")
        results.print_summary()

        if params.get('report', True) and HTMLReporter:
            report_file = logfile.replace('.log', '.html')
            reporter = HTMLReporter(filename=report_file)
            reporter.generate_report(results, mode="Uncertainty")
        
    except Exception as e:
        logger.critical(f"Execution Error: {e}")
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()

def execute_referral(params):
    logfile = params.get('logfile', 'roby_referral.log')
    logger = setup_logger(logfile)
    unc_method = params.get('unc_method', 'entropy')
    
    logger.info(f"STARTING REFERRAL: {unc_method}")
    print(f"\n--- REFERRAL LEARNING ---")

    try:
        classes = load_classes_robust(params['labelfile'])
        
        file_list = _prepare_file_list(params['inputpath'], int(params.get('max_samples', 0)), logger)
        if not file_list: return
        
        alteration = None
        if params.get('altname') and params.get('altparams'):
             alteration = load_alteration(params.get('module_name'), params['altname'], params['altparams'])
        
        env = None
        models_dir = params.get('models_dir')
        n_models = int(params.get('n_models', 0))
        input_shape_hint = params.get('input_shape', None)
        device = params.get('device', 'cpu')

        if models_dir and n_models > 0:
            # Carica ensemble
            ensemble = EnsembleTools.load_models(n_models, models_dir, ensemble_models=True,
                                                 input_shape=input_shape_hint, device=device)
            h, w, c = get_model_input_shape(ensemble[0])
            preprocess_f = create_preprocessing_f(h, w, c)
            env = EnvironmentRTest.EnvironmentRTest(ensemble_models=ensemble, file_list=file_list, 
                                                     classes=classes, labeler_f=labeler, 
                                                     reader_f=reader, preprocess_f=preprocess_f)
        else:
            # Carica modello singolo
            modelpath = params['modelpath']
            model = _load_model_unified(modelpath, input_shape=input_shape_hint, device=device)
            
            h, w, c = get_model_input_shape(model)
            preprocess_f = create_preprocessing_f(h, w, c)
            env = EnvironmentRTest.EnvironmentRTest(model=model, file_list=file_list, 
                                                     classes=classes, labeler_f=labeler, 
                                                     reader_f=reader, preprocess_f=preprocess_f)
        
        results = ReferralNN.referral_test(
            env, alteration, 
            float(params.get('altlevel', 0.0)), 
            unc_method, 
            int(params.get('n_mc_samples', 10)), 
            int(params.get('steps', 20))
        )
        
        results.print_summary()
        viz_ref.plot_referral_curve(results)
        log_summary(logger, results, mode="Referral")

        if params.get('report', True) and HTMLReporter:
            report_file = logfile.replace('.log', '.html')
            reporter = HTMLReporter(filename=report_file)
            reporter.generate_report(results, mode="Referral")
        
    except Exception as e:
        logger.critical(f"Execution Error: {e}")
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()


# --- CLI WRAPPERS ---

def _parse_input_shape(shape_str):
    """Parse input shape string 'h,w,c' to tuple (h, w, c)"""
    if shape_str:
        try:
            parts = [int(x.strip()) for x in shape_str.split(',')]
            if len(parts) == 3:
                return tuple(parts)
        except:
            pass
    return None

@click.group()
def cli():
    """Roby: Neural Network Robustness and Uncertainty Analysis Tool."""
    pass

@cli.command('robustness')
@click.option('--modelpath', required=True, type=str)
@click.option('--inputpath', required=True, type=str)
@click.option('--labelfile', required=True, type=str)
@click.option('--altname', required=True, type=str)
@click.option('--altparams', required=True, type=str)
@click.option('--npoints', required=True, type=int)
@click.option('--theta', required=True, type=float)
@click.option('--module_name', default="roby.Alterations")
@click.option('--logfile', default="roby_robustness.log")
@click.option('--debug/--no-debug', default=True)
@click.option('--report/--no-report', default=True, help="Genera report HTML")
@click.option('--max_samples', default=0, type=int, help="Limita il numero di immagini da testare (0=tutte)")
@click.option('--input_shape', default=None, type=str, help="Input shape per PyTorch: 'h,w,c' (es. '28,28,1')")
@click.option('--device', default='cpu', type=str, help="Device PyTorch: 'cpu' o 'cuda'")
def run_robustness(**kwargs):
    # Parse input_shape
    kwargs['input_shape'] = _parse_input_shape(kwargs.get('input_shape'))
    execute_robustness(kwargs)

@cli.command('uncertainty')
@click.option('--modelpath', required=False, type=str)
@click.option('--models_dir', required=False, type=str)
@click.option('--n_models', required=False, default=0, type=int)
@click.option('--inputpath', required=True, type=str)
@click.option('--labelfile', required=True, type=str)
@click.option('--altname', required=True, type=str)
@click.option('--altparams', required=True, type=str)
@click.option('--npoints', required=True, type=int)
@click.option('--unc_method', default='entropy', type=click.Choice(['entropy', 'confidence', 'mi', 'mean_entropy']))
@click.option('--threshold', default=0.5, type=float)
@click.option('--n_mc_samples', default=10, type=int)
@click.option('--module_name', default="roby.Alterations")
@click.option('--logfile', default="roby_uncertainty.log")
@click.option('--report/--no-report', default=True, help="Genera report HTML")
@click.option('--max_samples', default=0, type=int, help="Limita il numero di immagini da testare (0=tutte)")
@click.option('--input_shape', default=None, type=str, help="Input shape per PyTorch: 'h,w,c' (es. '28,28,1')")
@click.option('--device', default='cpu', type=str, help="Device PyTorch: 'cpu' o 'cuda'")
def run_uncertainty(**kwargs):
    # Parse input_shape
    kwargs['input_shape'] = _parse_input_shape(kwargs.get('input_shape'))
    execute_uncertainty(kwargs)

@cli.command('referral')
@click.option('--modelpath', required=False, type=str)
@click.option('--models_dir', required=False, type=str)
@click.option('--n_models', required=False, default=0, type=int)
@click.option('--inputpath', required=True, type=str)
@click.option('--labelfile', required=True, type=str)
@click.option('--altname', required=False, type=str)
@click.option('--altparams', required=False, type=str)
@click.option('--altlevel', required=False, default=0.0, type=float)
@click.option('--unc_method', default='entropy')
@click.option('--n_mc_samples', default=10, type=int)
@click.option('--steps', default=20, type=int)
@click.option('--module_name', default="roby.Alterations")
@click.option('--logfile', default="roby_referral.log")
@click.option('--report/--no-report', default=True, help="Genera report HTML")
@click.option('--max_samples', default=0, type=int, help="Limita il numero di immagini da testare (0=tutte)")
@click.option('--input_shape', default=None, type=str, help="Input shape per PyTorch: 'h,w,c' (es. '28,28,1')")
@click.option('--device', default='cpu', type=str, help="Device PyTorch: 'cpu' o 'cuda'")
def run_referral(**kwargs):
    # Parse input_shape
    kwargs['input_shape'] = _parse_input_shape(kwargs.get('input_shape'))
    execute_referral(kwargs)

@cli.command('from_config')
@click.argument('config_file', type=click.Path(exists=True))
def run_from_config(config_file):
    """Esegue un test leggendo la configurazione da un file JSON."""
    print(f"Caricamento configurazione da: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Errore nella lettura del JSON: {e}")
        return

    mode = config.get('mode', '').lower()
    params = config.get('parameters', {})
    
    # Merge parametri comuni se esistono
    common = config.get('common', {})
    if common:
        params = {**common, **params}
    
    # Default report e max_samples
    if 'report' not in params: params['report'] = True
    if 'max_samples' not in params: params['max_samples'] = 0
    
    # Parse input_shape se è una lista [h, w, c]
    if 'input_shape' in params and isinstance(params['input_shape'], list):
        if len(params['input_shape']) == 3:
            params['input_shape'] = tuple(params['input_shape'])

    if mode == 'robustness':
        execute_robustness(params)
    elif mode == 'uncertainty':
        execute_uncertainty(params)
    elif mode == 'referral':
        execute_referral(params)
    else:
        print(f"Mode '{mode}' non riconosciuto. Usa: 'robustness', 'uncertainty' o 'referral'.")

if __name__ == '__main__':
    cli()
