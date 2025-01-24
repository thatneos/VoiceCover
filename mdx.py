import gc
import hashlib
import os
import queue
import threading
import warnings
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm
from audio_separator.separator import Separator
import re
import random


def run_uvr(model_params, output_dir, model_name, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process the audio
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    output_names = {
        "Vocals": f"{base_name}_Vocals.wav",
        "Instrumental": f"{base_name}_Instrumental.wav",
    }
    vocals_file = f"{base_name}_Vocals.wav"
    instrumental_file = f"{base_name}_Instrumental.wav"

    separator = Separator(output_dir=output_dir)
    separator.load_model(model_filename=model_name, mdx_enable_denoise=denoise)
    output_files = separator.separate(filename, output_names)


    main_filepath = None
    invert_filepath = None

    if not exclude_main:
        main_filepath = os.path.join(output_dir, vocals_file)
        
    if not exclude_inversion:
        invert_filepath = os.path.join(output_dir, instrumental_file)
        
    if not keep_orig:
        os.remove(filename)

    return main_filepath, invert_filepath
