#https://github.com/coqui-ai/TTS/tree/dev#install-tts
import numpy as np
import torch
from TTS.api import TTS
from utils import genDataFrame
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import soundfile as sf


#get device and paths
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/mod_asvspoof2019"
dataset_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

#initialize models for VC
tts_VC = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=device)

#generate dataset and speaker list
speaker_list, df = genDataFrame(dataset_path, 0)

window_size = 1024
hop_length = 512
n_mels = 128
time_steps = 384

#generation loop

for speaker in speaker_list:

    #filter database by speaker
    temp = df[df["speaker"] == speaker]
    # pass all the files from the speaker for context to knn_vc, extract the single file path,
    # perform asr, perform tts, perform vc, save results, delete tts files
    # knn_vc reference file
    knn_reference = knn_vc.get_matching_set(temp["audio_path"].values.tolist())

    #for loop to pass every file path in temp
    for _, row in temp.iterrows():

        original, s = librosa.load(os.path.join(base_path, row["audio_path"]), sr=16000)
        sf.write(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data', row["filename"] + ".wav"), original,
                 samplerate=s)
        tts_VC.voice_conversion_to_file(source_wav= row["audio_path"], target_wav= row["audio_path"], file_path=os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data', row["filename"] + "_VC_1.wav"))
        audio1, s = librosa.load(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data', row["filename"] + "_VC_1.wav"), sr=16000)
        knn_original = knn_vc.get_features(row["audio_path"])
        audio2 = knn_vc.match(knn_original, knn_reference, topk=4)
        sf.write(os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data', row["filename"] + "_VC_2.wav"), audio2,
                 samplerate=s)


        window = np.hanning(window_size)
        stft1 = librosa.core.spectrum.stft(np.asarray(original), n_fft=window_size, hop_length=hop_length, window=window)
        stft2 = librosa.core.spectrum.stft(audio1, n_fft=window_size, hop_length=hop_length, window=window)
        stft3 = librosa.core.spectrum.stft(audio2.cpu().numpy(), n_fft=window_size, hop_length=hop_length, window=window)
        out1 = 2 * np.abs(stft1) / np.sum(window)
        out2 = 2 * np.abs(stft2) / np.sum(window)
        out3 = 2 * np.abs(stft3) / np.sum(window)
        diff1 = out1[512,136]-out2
        diff2 = out1[512,136]-out3

        plt.figure(figsize=(12, 4), dpi=256)
        ax = plt.axes()
        librosa.display.specshow(librosa.amplitude_to_db(diff1), y_axis='log', x_axis='time', sr=s)
        plt.savefig('/nas/home/aorsatti/Pycharm/Tesi/data/spectrogram1db.png', bbox_inches='tight', transparent=True, pad_inches=0.0)
        librosa.display.specshow(librosa.amplitude_to_db(diff2), y_axis='log', x_axis='time', sr=s)
        plt.savefig('/nas/home/aorsatti/Pycharm/Tesi/data/spectrogram2db.png', bbox_inches='tight', transparent=True, pad_inches=0.0)
        print('Mse 1 is : ' + str(((original[min(len(original),len(audio1))-1]-audio1[min(len(original),len(audio1))-1])**2).mean()))
        print('Mse 2 is : ' + str(((original[min(len(original),len(audio2))-1]-audio2[min(len(original),len(audio2))-1].cpu().numpy())**2).mean()))

print("Done!")



