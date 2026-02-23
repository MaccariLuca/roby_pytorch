import setuptools

setuptools.setup(
    name="roby",
    version="0.0.5",  
    author="University of Bergamo",
    author_email="andrea.bombarda@unibg.it",
    description="A general framework to analyse the robustness of a " +
                "Neural Network",
    packages=setuptools.find_packages(),
    python_requires='>=3.9', #per python piu recente
    install_requires=[
        'numpy>=1.26.0',
        'opencv-python>=4.0',  # Il nuovo nome pacchetto è opencv-python
        'Pillow>=10.0.0',      # Aggiornato a versione piu recente
        'tensorflow>=2.15.0',  # TensorFlow include Keras
        'scikit-learn>=1.3.0',
        'PyDrive>=1.3.1',
        'scipy>=1.10.0',
        'oauth2client>=4.1.3', # libreria datata, ma mantenuta perchè non trovo sostituto
        'matplotlib>=3.7.0',
        'protobuf>=3.20.3',
        'typing-extensions>=4.0.0; python_version < "3.9"', # typing è già incluso nelle nuove versioni
        'imutils>=0.5.2',
        'sympy>=1.12'
    ])