#https://github.com/coqui-ai/TTS/tree/dev#install-tts
import numpy as np
import torch
from TTS.api import TTS
from utils import genDataFrame, checkCharacters
import whisper
import soundfile as sf
import pandas as pd
import os
import librosa
from tqdm import tqdm
import pdb

#get device and paths
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/modified_asvspoof2019"
dataset_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
models_tts = ['tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/jenny/jenny', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/tacotron2-DCA']
#initialize models for ASR and VC
model = whisper.load_model("base.en").to(device)
tts_VC = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=device)

#generate dataset and speaker list
speaker_list, df = genDataFrame(dataset_path)

#generate empty dataframe to store new data
if not os.path.isfile('/nas/home/aorsatti/Pycharm/Tesi/data/asvspoof2019_eval_modified.txt'):
    df_vc = pd.DataFrame(columns = ['speaker', 'filename', 'label', 'original_filename', 'audio_path'])
    df_vc = pd.concat([df_vc, df], axis=0)
else:
    df_vc = pd.read_csv('/nas/home/aorsatti/Pycharm/Tesi/data/asvspoof2019_eval_modified.txt', sep=' ')

#bad_chars = checkCharacters(df)
bad_chars = ['-', '&', ')', '�', '–', '%', '{', '—', '=', '"']

#generation loop
for i in range(len(models_tts)):
    #initialize the model for TTS
    tts = TTS(model_name=models_tts[i], progress_bar=False).to(device)

    for speaker in speaker_list:

        #filter database by speaker
        temp = df[df["speaker"] == speaker]
        # pass all the files from the speaker for context to knn_vc, extract the single file path,
        # perform asr, perform tts, perform vc, save results, delete tts files
        # knn_vc reference file
        knn_reference = knn_vc.get_matching_set(temp["audio_path"].values.tolist())

        #for loop to pass every file path in temp
        for _, row in tqdm(temp.iterrows(), total=len(temp)):

            if not os.path.isfile(os.path.join(base_path, row["filename"] + "_TTS_" + str(i+1) + "_VC_1.wav")):
                
                # Whisper
                result = model.transcribe(row["audio_path"])
                text = result["text"]
                if text == '':
                    text = 'hi'
                for k in range(len(bad_chars)):
                    text = text.replace(bad_chars[k], '')

                # try:
                tts.tts_to_file(text=text, file_path=os.path.join(base_path, "temp.wav"))

                tts_VC.voice_conversion_to_file(source_wav= os.path.join(base_path, "temp.wav"), target_wav= row["audio_path"], file_path=os.path.join(base_path, row["filename"] + "_TTS_" + str(i+1) + "_VC_1.wav"))
                audio, s = librosa.load(os.path.join(base_path, row["filename"] + "_TTS_" + str(i+1) + "_VC_1.wav"), sr=16000)
                sf.write(os.path.join(base_path, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav"), np.asarray(audio),
                         samplerate=s)

                knn_original = knn_vc.get_features(os.path.join(base_path, "temp.wav"))
                audio = knn_vc.match(knn_original, knn_reference, topk=4)
                sf.write(os.path.join(base_path, row["filename"] + "_TTS_" + str(i+1) + "_VC_2.wav"), audio, samplerate=16000)
                df_vc = df_vc.append({'speaker': speaker, 'filename': row["filename"] + "_TTS_" + str(i+1) +"_VC_2", 'label': 'spoof', 'original_filename': row["filename"], 'audio_path': os.path.join(base_path, row["filename"] + "_TTS_" + str(i+1) + "_VC_2.wav")}, ignore_index=True)
                df_vc = df_vc.append(
                    {'speaker': speaker, 'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_1", 'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join(base_path, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav")},
                    ignore_index=True)
                df_vc.to_csv('/nas/home/aorsatti/Pycharm/Tesi/data/asvspoof2019_eval_modified.txt')
                # except:
                #     print()
                #     pdb.set_trace()


df_vc = df_vc.sort_values('filename')
df_vc.to_csv('/nas/home/aorsatti/Pycharm/Tesi/data/asvspoof2019_eval_modified.txt')