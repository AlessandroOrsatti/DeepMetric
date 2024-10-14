
import numpy as np
import torch
from TTS.api import TTS
from utils import generationDataFrameV2, checkCharacters, generateCsvV2
import whisper
import soundfile as sf
import pandas as pd
import os
import librosa
from tqdm import tqdm
import parameters as param

#get device and paths
device = "cuda" if torch.cuda.is_available() else "cpu"

#initialize models for ASR and VC
model = whisper.load_model("base.en").to(device)
tts_VC = TTS(model_name=param.model_vc, progress_bar=False).to(device)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=device)

#generate dataset and speaker list
df = generationDataFrameV2()

#bad_chars = checkCharacters(df)
bad_chars = ['-', '&', ')', '�', '–', '%', '{', '—', '=', '"']
df_vc = pd.DataFrame(columns=['filename', 'label', 'original_filename', 'audio_path'])
df_vc = pd.concat([df_vc, df], axis=0)

#generation loop
for i in range(len(param.models_tts)):

    tts = TTS(model_name=param.models_tts[i], progress_bar=False).to(device)
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        if not os.path.isfile(
                os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                             row["filename"] + "_TTS_" + str(
                                 i + 1) + ".wav")):
            result = model.transcribe(row["audio_path"])
            text = result["text"]
            if '...' in text:
                text = text.replace('...', '')
            if text == '':
                text = 'hi'
            for k in range(len(bad_chars)):
                text = text.replace(bad_chars[k], '')

            tts.tts_to_file(text=text, file_path=os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + ".wav"))
            tts_VC.voice_conversion_to_file(source_wav=os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + ".wav"),
                                                            target_wav=row["audio_path"],
                                                            file_path=os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + "_VC_1.wav"))
            audio, s = librosa.load(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + "_VC_1.wav"),sr=param.sr)
            sf.write(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + "_VC_1.wav"), np.asarray(audio), samplerate=s)

            knn_original = knn_vc.get_features(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + ".wav"))
            knn_reference = knn_vc.get_matching_set([row["audio_path"]])
            audio = knn_vc.match(knn_original, knn_reference, topk=4)
            sf.write(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                                                   row["filename"] + "_TTS_" + str(
                                                                                       i + 1) + "_VC_2.wav"),audio, samplerate=param.sr)
            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1),
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2', row["filename"] + "_TTS_" + str(i + 1) + ".wav")},
                ignore_index=True)

            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_1", 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2', row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav")},
                ignore_index=True)

            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_2", 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2', row["filename"] + "_TTS_" + str(i + 1) + "_VC_2.wav")},
                ignore_index=True)

            df_vc.to_csv(os.path.join(param.csv_dir, 'modified_dataset2.csv'))


df_vc = df_vc.sort_values('filename')
df_vc.to_csv(os.path.join(param.csv_dir, 'modified_dataset2.csv'))