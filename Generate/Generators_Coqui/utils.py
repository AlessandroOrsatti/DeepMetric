from vosk import Model, KaldiRecognizer
import json
import wave
import pandas as pd
import os
import whisper
import torch
from tqdm import tqdm

vosk_models_path = "/nas/home/aorsatti/Pycharm/Tesi/data/vosk_models/"
vosk_models = ["vosk-model-en-us-0.22-lgraph", "vosk-model-en-us-0.42-gigaspeech"]
def get_text_from_voice(filename, large=False):

    wf = wave.open(filename, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        exit(1)

    if(large):
        model = Model(vosk_models_path + vosk_models[1])
    else:
        model = Model(vosk_models_path + vosk_models[0])

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text_lst = []
    p_text_lst = []
    p_str = []
    len_p_str = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            text_lst.append(rec.Result())
            print(rec.Result())
        else:
            p_text_lst.append(rec.PartialResult())
            print(rec.PartialResult())

    if len(text_lst) != 0:
        jd = json.loads(text_lst[0])
        txt_str = jd["text"]

    elif len(p_text_lst) != 0:
        for i in range(0, len(p_text_lst)):
            temp_txt_dict = json.loads(p_text_lst[i])
            p_str.append(temp_txt_dict['partial'])

        len_p_str = [len(p_str[j]) for j in range(0, len(p_str))]
        max_val = max(len_p_str)
        indx = len_p_str.index(max_val)
        txt_str = p_str[indx]

    else:
        txt_str = ''

    return txt_str

def genDataFrame(label_path, truncate=2000):
    df = pd.read_csv(label_path, sep=' ', header=None).rename(columns={0:'speaker', 1:'filename', 2:'-', 3:'algorithm', 4:'label'})


    audio_dir = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'

    # MODIFIED------------------------------------
    # df = df.loc[df['label'] == 'bonafide']
    # df = df.reset_index(drop=True)
    # df = df.truncate(after=truncate)
    # df = df.sort_values('speaker')
    # df = df.drop(columns=['-', 'algorithm', 'label'])

    #---------------------------------------------
    df = df.drop(columns=['-', 'algorithm'])
    #df = df.truncate(after=truncate)
    df['original_filename'] = '-'
    df['audio_path'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x + '.flac'))

    speaker_list = df['speaker'].values.tolist()
    speaker_list = list(dict.fromkeys(speaker_list))


    return speaker_list, df

def checkCharacters(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base.en").to(device)
    special = []
    file = open('/nas/home/aorsatti/Pycharm/Tesi/data/badChars.txt', 'w')

    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = model.transcribe(row["audio_path"])
        text = result["text"]
        for i in text:
            if i.isalnum():
                continue
            else:
                special.append(i)
        special = list(dict.fromkeys(special))
    special.remove(' ','.',',','!','?', '"')

    mystring = ' '.join(map(str, special))
    file.write(mystring)

    return(special)
