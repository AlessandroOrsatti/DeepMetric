import os
import numpy as np

#costants
sr = 16000
eps = 1e-8
winLen = 1

#data
base_path = '/nas/home/aorsatti/Pycharm/Tesi'
base_storage_path = os.path.join(base_path, 'data')
models_dir = os.path.join(base_storage_path, 'models')
results_dir = os.path.join(base_storage_path, 'results')
embeddings_dir = os.path.join(base_storage_path, 'embeddings')
csv_dir = os.path.join(base_storage_path, 'csv')
mod_eval_dir = os.path.join(base_storage_path, 'modified_asvspoof2019')

#dataset
dataset_path = os.path.join('/nas/public/dataset/asvspoof2019', 'LA')
dataset_csv_path = os.path.join(dataset_path, 'ASVspoof2019_LA_cm_protocols')
train_dir = os.path.join(dataset_path, 'ASVspoof2019_LA_train', 'flac')
dev_dir = os.path.join(dataset_path, 'ASVspoof2019_LA_dev', 'flac')
eval_dir = os.path.join(dataset_path, 'ASVspoof2019_LA_eval', 'flac')

#vosk
vosk_models_path = os.path.join(base_storage_path, 'vosk_models')
vosk_models = ["vosk-model-en-us-0.22-lgraph", "vosk-model-en-us-0.42-gigaspeech"]

#dictionaries
generation_dict = {0:'speaker', 1:'filename', 2:'-', 3:'algorithm', 4:'label'}
generation_dict2 = {0:'speaker', 1:'filename', 2:'-', 3:'algorithm', 4:'label', 5:'TTS', 6:'VC'}
label_dict = {'bonafide': 0, 'spoof': 1}

#lists
test_dataframe = ['ID', 'original', 'label', 'prediction']

#tts
models_tts = ['tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/jenny/jenny', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/tacotron2-DCA']
model_vc = "voice_conversion_models/multilingual/vctk/freevc24"

