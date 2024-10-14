#https://github.com/coqui-ai/TTS/tree/dev#install-tts

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import torch
import string
from TTS.api import TTS
import soundfile as sf
import librosa


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/"
file_path = "/nas/home/aorsatti/benny/benny_vc2.wav"
fs = 16000

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

d = ModelDownloader()

speech2text = Speech2Text(
    **d.download_and_unpack("Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave"),
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)

speech, rate = sf.read(file_path)
if (rate!=fs):
    speech = librosa.resample(speech, orig_sr=rate, target_sr=fs)
nbests = speech2text(speech)
text, *_ = nbests[0]
text = text_normalizer(text)

# List available 🐸TTS models
# print(TTS().list_models())

# Tts and tts with vc
# num models selected: 7
models_tts = ['tts_models/en/ljspeech/tacotron2-DDC', 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/jenny/jenny']
for i in range(7):
    # Init TTS with the target model name
    tts = TTS(model_name=models_tts[i], progress_bar=True).to(device)

    # Run TTS
    tts.tts_to_file(text=text, file_path= base_path + str(i+1) + "_iwanttoridemybike.wav")
    #tts.tts_with_vc_to_file(text="I want to ride my bike.", speaker_wav="target/speaker.wav", file_path= base_path + str(i+1) + "_iwanttoridemybike_vc.wav")

# Tts multi-speaker and multi-lingual model
# num models selected: 3
# Init TTS
models_multi = ['tts_models/multilingual/multi-dataset/xtts_v2', 'tts_models/multilingual/multi-dataset/xtts_v1.1', 'tts_models/multilingual/multi-dataset/your_tts']
for i in range(3):
    tts = TTS(models_multi[i]).to(device)

    # Run TTS
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav= base_path + "iwanttoridemybike.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(text="Hello world!", speaker_wav= base_path + "iwanttoridemybike.wav", language="en", file_path= base_path + str(i+1) + "_multi.wav")


# Voice conversion
# num models selected:
models_vc = ["voice_conversion_models/multilingual/vctk/freevc24"]
for i in range(1):

    tts = TTS(model_name=models_vc[i], progress_bar=False).to("cuda")
    tts.voice_conversion_to_file(source_wav=..., target_wav=..., file_path= base_path + str(i+1) + "_vc.wav")


#tts --out_path /nas/home/aorsatti/benny/benny_vc.wav --model_name tts_models/en/ljspeech/tacotron2-DDC --source_wav /nas/home/aorsatti/Pycharm/Tesi/data/TTS_VC/1_iwanttoridemybike.wav --target_wav /nas/home/aorsatti/benny/benny_vc.wav


