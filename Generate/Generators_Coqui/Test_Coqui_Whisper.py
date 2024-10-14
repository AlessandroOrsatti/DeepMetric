#https://github.com/coqui-ai/TTS/tree/dev#install-tts
import time
import torch
import torchaudio
from TTS.api import TTS
from utils import get_text_from_voice
import whisper
import soundfile as sf


# List available 🐸TTS models
# print(TTS().list_models())

start = time.time()

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/"
file_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/iwanttoridemybike.wav"
vc_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/coqui_vosk/original.wav"
#vc_path ="/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0000.flac"
start_asr = time.time()

# Whisper
model = whisper.load_model("tiny.en").to(device)
result = model.transcribe(file_path)
text = result["text"]
print(text)

# Vosk
#text = get_text_from_voice(file_path, False)

print("ASR in: " + str(time.time()-start_asr))
start_tts = time.time()

# Tts and tts with vc
# num models selected: 7
models_tts = ['tts_models/multilingual/multi-dataset/bark', 'tts_models/en/ljspeech/tacotron2-DDC', 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/jenny/jenny']

tts_VC = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=device)

knn_reference = knn_vc.get_matching_set(["/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0000.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0001.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0002.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0003.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0004.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0005.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0006.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0007.flac", "/nas/public/dataset/LibriSpeech/train-clean-100/2136/5140/2136-5140-0008.flac"])

for i in range(len(models_tts)):
    # Init TTS with the target model name
    tts = TTS(model_name=models_tts[i], progress_bar=False).to(device)

    if (models_tts[i]=='tts_models/multilingual/multi-dataset/bark'):
        tts.tts_to_file(text=text, speaker_wav=file_path, file_path=base_path + str(i+1) + "_test_coqui.wav")
    else:
        tts.tts_to_file(text=text, file_path= base_path + str(i+1) + "_test_coqui.wav")
        tts_VC.voice_conversion_to_file(source_wav= base_path + str(i+1) + "_test_coqui.wav", target_wav= vc_path, file_path=base_path + str(i + 1) + "_test_coqui_vc.wav")
        knn_original = knn_vc.get_features(base_path + str(i+1) + "_test_coqui.wav")
        audio = knn_vc.match(knn_original, knn_reference, topk=4)
        sf.write(base_path + str(i + 1) + "_test_coqui_vc2.wav", audio, samplerate=16000)


print("TTS and VC in: " + str(time.time()-start_tts))
print("Finished in: " + str(time.time()-start))





