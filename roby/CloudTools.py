"""
The **CloudTools** module contains useful functions to be used on Google Colab
to perform robustness analysis on cloud based platform.

It offers the following functionalities:

- Load configurations from an `XML` file
- Grant the permission on a Google Drive account, where the dataset and model
  are stored
- Configure the environment either by starting from an `XML` file or directly
  giving the relevant links

@author: Andrea Bombarda
"""

from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# Manage xml and csv files
import xml.etree.ElementTree as Et

# roby tools
# RIMOSSO IMPORT GLOBALE PER EVITARE DIPENDENZE CIRCOLARI
# from roby.RobustnessNN import set_classes 
from roby.EnvironmentRTest import EnvironmentRTest
from typing import Callable, List, Any
import numpy as np

# --- Gestione Import Opzionali per Google Colab ---
try:
    from google.colab import files   # type: ignore
    from pydrive.auth import GoogleAuth   # type: ignore
    from pydrive.drive import GoogleDrive   # type: ignore
    from google.colab import auth
    from oauth2client.client import GoogleCredentials   # type: ignore
    COLAB_AVAILABLE = True
except ImportError:
    # Se siamo su PC locale, definiamo classi dummy per evitare errori di import
    COLAB_AVAILABLE = False
    class GoogleDrive: pass
    class GoogleAuth: pass
    files = None
    auth = None
    GoogleCredentials = None

def _check_colab():
    if not COLAB_AVAILABLE:
        raise ImportError("Google Colab modules not found. This function only works on Colab.")

def upload_config() -> Et.Element:
    """
    Uploading and parsing of configuration file
    """
    _check_colab()
    uploaded = files.upload()

    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))

    tree = Et.parse('config.xml')
    root = tree.getroot()
    return root


def authenticate() -> GoogleDrive:
    """
    Mounts Google Drive into the virtual machine
    """
    _check_colab()
    auth.authenticate_user()
    g_auth = GoogleAuth()
    g_auth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(g_auth)
    return drive


def process_config_from_xml(root: Et.Element, drive: GoogleDrive,
                            labeler_f: Callable[[np.ndarray], str]=None,
                            pre_processing_f:
                            Callable[[np.ndarray], np.ndarray]=None,
                            reader_f:
                            Callable[[str], np.ndarray]=None) \
                            -> EnvironmentRTest:
    """
    Loads the configuration data from an XML file and builds the environment
    """
    model_link = None
    link_dataset_dir = None
    classes_link = None
    labels_link = None
    for elem in root.iter('ModelLink'):
        model_link = elem.text
    for elem in root.iter('DatasetLink'):
        link_dataset_dir = elem.text
    for elem in root.iter('Classes'):
        classes_link = elem.text
    for elem in root.iter('Labels'):
        labels_link = elem.text

    if model_link is None:
        raise ValueError("No model link found in the configuration file")
    if link_dataset_dir is None:
        raise ValueError("No dataset link found in the configuration file")
    if classes_link is None:
        raise ValueError("No classes link found in the configuration file")

    return process_config(model_link, link_dataset_dir, classes_link, drive,
                          labels_link, labeler_f,
                          pre_processing_f, reader_f)


def load_model_from_url(model_link: str, drive: GoogleDrive) -> Model:
    """
    Loads the model from a Google Drive URL
    """
    # Upload the Model
    id_val = model_link.split('/')[-1]
    downloaded = drive.CreateFile({'id': id_val})
    downloaded.GetContentFile('Model.model')
    model = load_model('Model.model')
    print('Model Uploaded')
    return model


def load_dataset_from_url(link_dataset_dir: str,
                          drive: GoogleDrive) -> List[str]:
    """
    Loads the input test dataset from a Google Drive URL
    """
    print('Uploading Dataset...')
    cont = 0
    id_val = link_dataset_dir.split('/')[-1]
    query = "'" + str(id_val) + "'" + ' in parents and trashed=false'
    file_list = drive.ListFile({'q': str(query)}).GetList()
    temp_list = []
    for file1 in file_list:
        file2 = drive.CreateFile({'id': file1['id']})
        file2.GetContentFile(file1['title'])
        temp_list.append(file1['title'])
        cont += 1
    file_list = temp_list
    print('Dataset Uploaded')
    return file_list


def load_classes_from_url(classes_link: str,
                          drive: GoogleDrive) -> List[str]:
    """
    Loads the classes from a Google Drive URL
    """
    # IMPORT LAZY: Risolve la dipendenza circolare importando solo quando serve
    from roby.RobustnessNN import set_classes

    id_val = classes_link.split('/')[-1]
    downloaded = drive.CreateFile({'id': id_val})
    downloaded.GetContentFile('Classes.csv')
    classes = set_classes('Classes.csv')
    print('Classes Uploaded')
    return classes


def load_labels_from_url(label_list_link: str,
                         drive: GoogleDrive) -> List[str]:
    """
    Loads the labels from a Google Drive URL
    """
    # IMPORT LAZY
    from roby.RobustnessNN import set_classes

    id_val = label_list_link.split('/')[-1]
    downloaded = drive.CreateFile({'id': id_val})
    downloaded.GetContentFile('Labels.csv')
    labels = set_classes('Labels.csv')
    print('Labels Uploaded')
    return labels


def process_config(model_link: str, link_dataset_dir: str, classes_link: str,
                   drive: GoogleDrive,
                   label_list_link: str=None,
                   labeler_f: Callable[[np.ndarray], str]=None,
                   pre_processing_f: Callable[[np.ndarray], np.ndarray]=None,
                   reader_f: Callable[[str], np.ndarray]=None) \
                   -> EnvironmentRTest:
    """
    Loads the configuration data from parameters and builds the enviroment
    """
    # Upload the Model
    model = load_model_from_url(model_link, drive)
    # Upload Dataset
    file_list = load_dataset_from_url(link_dataset_dir, drive)
    # Upload Classes
    classes = load_classes_from_url(classes_link, drive)
    # Upload Labels
    if label_list_link is not None:
        labels = load_labels_from_url(label_list_link, drive)
    else:
        labels = None   # type: ignore

    # Create the object wit all these variables saved
    environment = EnvironmentRTest(model, file_list, classes, labels,
                                   labeler_f, pre_processing_f, reader_f)
    return environment