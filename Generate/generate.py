
import numpy as np
import torch
from TTS.api import TTS
from utils import generationDataFrame, checkCharacters, generateCsv
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
speaker_list, df = generationDataFrame(os.path.join(param.dataset_csv_path, 'ASVspoof2019.LA.cm.eval.trl.txt'))

#bad_chars = checkCharacters(df)
bad_chars = ['-', '&', ')', '�', '–', '%', '{', '—', '=', '"']

#generation loop
for i in range(len(param.models_tts)):
    # initialize the model for TTS
    tts = TTS(model_name=param.models_tts[i], progress_bar=False).to(device)

    for speaker in tqdm(speaker_list, total=len(speaker_list)):

        # filter database by speaker
        temp = df[df["speaker"] == speaker]
        # pass all the files from the speaker for context to knn_vc, extract the single file path,
        # perform asr, perform tts, perform vc, save results, delete tts files
        # knn_vc reference file
        knn_reference = knn_vc.get_matching_set(temp["audio_path"].values.tolist())

        # for loop to pass every file path in temp
        for _, row in temp.iterrows():

            if not os.path.isfile(
                    os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav")):

                # Whisper
                result = model.transcribe(row["audio_path"])
                text = result["text"]
                if '...' in text:
                    text = text.replace('...', '')
                if text == '':
                    text = 'hi'
                for k in range(len(bad_chars)):
                    text = text.replace(bad_chars[k], '')

                tts.tts_to_file(text=text, file_path=os.path.join(param.mod_eval_dir,
                                                                           row["filename"] + "_TTS_" + str(
                                                                               i + 1) + ".wav"))
                tts_VC.voice_conversion_to_file(source_wav=os.path.join(param.mod_eval_dir, "temp.wav"),
                                                    target_wav=row["audio_path"],
                                                    file_path=os.path.join(param.mod_eval_dir,
                                                                           row["filename"] + "_TTS_" + str(
                                                                               i + 1) + "_VC_1.wav"))
                audio, s = librosa.load(os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav"),sr=param.sr)
                sf.write(os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav"), np.asarray(audio), samplerate=s)

                knn_original = knn_vc.get_features(os.path.join(param.mod_eval_dir, "temp.wav"))
                audio = knn_vc.match(knn_original, knn_reference, topk=4)
                sf.write(os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_2.wav"),audio, samplerate=param.sr)

generateCsv(df, os.path.join(param.csv_dir, 'asvspoof2019_eval_modified.csv'))
