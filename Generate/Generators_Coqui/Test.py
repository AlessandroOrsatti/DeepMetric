

import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
tts.tts_to_file(text="Questa sera siete invitati a casa mia per cena, froci!", speaker_wav= "/nas/home/aorsatti/benny/benny.wav", language="it", file_path= "/nas/home/aorsatti/benny/benny_vc.wav")

tts = TTS('tts_models/en/ljspeech/tacotron2-DDC').to(device)
tts.tts_with_vc_to_file(
    "Tonight you are invited as a guest in my house.",
    speaker_wav="/nas/home/aorsatti/benny/benny.wav",
    file_path="/nas/home/aorsatti/benny/benny_vc2.wav"
)