#https://github.com/coqui-ai/TTS/tree/dev#install-tts
import time
import torch
from TTS.api import TTS
from utils import get_text_from_voice
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import whisper



# List available 🐸TTS models
# print(TTS().list_models())

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

hl = 512 # number of samples per time-step in spectrogram
hi = 128 # Height of image
wi = 384 # Width of image

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/"
file_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/iwanttoridemybike.wav"
vc_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/original.wav"

start = time.time()

model = whisper.load_model("base").to(device)
result = model.transcribe(file_path)
text = result["text"]
print(text)

print("ASR in: " + str(time.time()-start))


# Tts and tts with vc
# num models selected: 7
models_tts = ['tts_models/en/ljspeech/tacotron2-DDC', 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/jenny/jenny']
models_vc = ["voice_conversion_models/multilingual/vctk/freevc24"]


tts_VC = TTS(model_name=models_vc[0], progress_bar=False).to(device)

for i in range(len(models_tts)):
    start = time.time()

    # Init TTS with the target model name
    tts = TTS(model_name=models_tts[i], progress_bar=False).to(device)

    # Run TTS
    tts.tts_to_file(text=text, file_path= base_path + str(i+1) + "_test_coqui.wav")
    tts_VC.voice_conversion_to_file(source_wav= base_path + str(i+1) + "_test_coqui.wav", target_wav= vc_path, file_path=base_path + str(i + 1) + "_test_coqui_vc.wav")
    print("Model " + str(i+1) + " TTS in: " + str(time.time() - start))

    # Plot and save image for TTS
    y, sr = librosa.load(base_path + str(i+1) + "_test_coqui.wav")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hi, fmax=8000, hop_length=hl)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    img2 = librosa.display.waveshow(y, sr=sr)

    plt.savefig(base_path + str(i + 1) + "_wf_out.png")
    plt.show()
    plt.close()


    # Plot and save image for TTS and VC

    y, sr = librosa.load(base_path + str(i+1) + "_test_coqui_vc.wav")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hi, fmax=8000,hop_length=hl)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    img2 = librosa.display.waveshow(y, sr=sr)

    plt.savefig(base_path + str(i+1) + "_wf_vc_out.png")
    plt.show()
    plt.close()





