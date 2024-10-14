#https://github.com/coqui-ai/TTS/tree/dev#install-tts
import time
import torch
from TTS.api import TTS
from utils import get_text_from_voice

# List available 🐸TTS models
# print(TTS().list_models())

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/"
file_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/iwanttoridemybike.wav"

start = time.time()

text = get_text_from_voice(file_path, False)

print("ASR in: " + str(time.time()-start))
start = time.time()

# Tts and tts with vc
# num models selected: 7
models_tts = ['tts_models/en/ljspeech/tacotron2-DDC', 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/jenny/jenny']
models_vc = ["voice_conversion_models/multilingual/vctk/freevc24"]


tts_VC = TTS(model_name=models_vc[0], progress_bar=False).to(device)

for i in range(len(models_tts)):
    # Init TTS with the target model name
    tts = TTS(model_name=models_tts[i], progress_bar=False).to(device)

    # Run TTS
    tts.tts_to_file(text=text, file_path= base_path + str(i+1) + "_test_coqui.wav")
    tts_VC.voice_conversion_to_file(source_wav= base_path + str(i+1) + "_test_coqui.wav", target_wav= file_path, file_path=base_path + str(i + 1) + "_test_coqui_vc.wav")
    #tts.tts_with_vc_to_file(text=text, speaker_wav=file_path, file_path= base_path + str(i + 1) + "_test_coqui_vc_int.wav")

print("TTS and VC in: " + str(time.time()-start))

# Tts multi-speaker and multi-lingual model
# num models selected: 3
# models_multi = ['tts_models/multilingual/multi-dataset/xtts_v2', 'tts_models/multilingual/multi-dataset/xtts_v1.1']
# for i in range(len(models_multi)):
#    tts = TTS(model_name=models_multi[i], progress_bar=False).to(device)

    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav= base_path + "iwanttoridemybike.wav", language="en")
    # Text to speech to a file
#    tts.tts_to_file(text=text, speaker_wav= file_path, language="en", file_path= base_path + str(i+1) + "_test_coqui_multi.wav")



