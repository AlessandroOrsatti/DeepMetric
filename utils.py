import os
import numpy as np
from torch import Tensor
from torch import nn
import librosa
from torch.utils.data import Dataset
import pandas as pd
import random
from src.torch_utils import set_gpu
from vosk import Model, KaldiRecognizer
import json
import wave
import pandas as pd
import scipy
import soundfile as sf
import whisper
import torch
from tqdm import tqdm
import parameters as param
from scipy.spatial import distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from os import listdir
from os.path import isfile, join


#Tell "from utils import *" what to import
__all__ = [
    "set_gpu",
]


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def pad(x, max_len):

    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def genDataFrame(label_path, is_train=False, is_eval=False, is_mod=False):

    if (is_train):
        df = pd.read_csv(label_path, sep=' ', header=None).rename(
            columns=param.generation_dict)
        df['label_bin'] = df['label'].map(param.label_dict)
        if 'train' in label_path:
            df['audio_path'] = df['filename'].apply(lambda x: os.path.join(param.train_dir, x + '.flac'))
        else:
            df['audio_path'] = df['filename'].apply(lambda x: os.path.join(param.dev_dir, x + '.flac'))

        file_list = df['filename'].values.tolist()
        speaker_dict = pd.Series(df.speaker.values, index=df.filename).to_dict()
        path_dict = pd.Series(df.audio_path.values, index=df.filename).to_dict()
        label_dict = pd.Series(df.label_bin.values, index=df.filename).to_dict()

        return label_dict, file_list, path_dict, speaker_dict

    elif (is_eval):
        if (is_mod==False):
            df = pd.read_csv(label_path, sep=' ', header=None)
            df = df.rename(
            columns=param.generation_dict2)
            df['audio_path'] = df['filename'].apply(lambda x: os.path.join(param.eval_dir, x + '.flac'))
            df['label_bin'] = df['label'].map(param.label_dict)
            file_list = df['filename'].values.tolist()
            path_dict = pd.Series(df.audio_path.values, index=df.filename).to_dict()
            label_dict = pd.Series(df.label_bin.values, index=df.filename).to_dict()
            return label_dict, file_list, path_dict
        else:
            df = pd.read_csv(label_path, sep=',', header=0)
            df['label_bin'] = df['label'].map({'bonafide': 0, 'spoof': 1})
            file_list = df['filename'].values.tolist()
            path_dict = pd.Series(df.audio_path.values, index=df.filename).to_dict()
            label_dict = pd.Series(df.label_bin.values, index=df.filename).to_dict()
            original_dict = pd.Series(df.original_filename.values, index=df.filename).to_dict()
            return label_dict, file_list, path_dict, original_dict

def generationDataFrame(label_path, is_truncate=False, truncate=100):
    df = pd.read_csv(label_path, sep=' ', header=None).rename(columns=param.generation_dict)

    # MODIFIED------------------------------------
    # df = df.loc[df['label'] == 'bonafide']
    # df = df.reset_index(drop=True)
    # df = df.truncate(after=truncate)
    # df = df.sort_values('speaker')
    # df = df.drop(columns=['-', 'algorithm', 'label'])

    #---------------------------------------------

    if is_truncate:
        df = df.truncate(after=truncate)

    df = df.drop(columns=['-', 'algorithm'])
    df['original_filename'] = '-'
    df['audio_path'] = df['filename'].apply(lambda x: os.path.join(param.eval_dir, x + '.flac'))

    speaker_list = df['speaker'].values.tolist()
    speaker_list = list(dict.fromkeys(speaker_list))


    return speaker_list, df

def generationDataFrameV2():
    path_real = '/nas/home/dsalvi/fake_or_real_FOR/for-original-wav/testing/real'
    path_fake = '/nas/home/dsalvi/fake_or_real_FOR/for-original-wav/testing/fake'
    df = pd.DataFrame(columns=['filename', 'label', 'original_filename', 'audio_path'])

    list_real = [f for f in listdir(path_real) if isfile(join(path_real, f))]
    list_fake = [f for f in listdir(path_fake) if isfile(join(path_fake, f))]

    for i in range(len(list_real)):
        df = df.append(
            {'filename': str(i + 1),
             'label': 'bonafide',
             'original_filename': '-',
             'audio_path': os.path.join(path_real, list_real[i] )},
            ignore_index=True)
    for i in range(len(list_fake)):
        df = df.append(
            {'filename': str(len(list_real) +i + 1),
             'label': 'spoof',
             'original_filename': '-',
             'audio_path': os.path.join(path_fake, list_fake[i] )},
            ignore_index=True)
    return df


class LoadTrainData(Dataset):
    def __init__(self, file_IDs, labels, audio_path, win_len):
        self.file_IDs = file_IDs
        self.labels = labels
        self.audio_path = audio_path
        self.win_len = win_len

    def __getitem__(self, index):

        if index % 2 == 0:
            X_id = random.choice(list({key : val for key, val in self.labels.items()
                   if val == 0}))
            y = 0
        else:
            X_id = random.choice(list({key : val for key, val in self.labels.items()
                   if val == 1}))
            y = 1

        X, fs = librosa.load(self.audio_path[X_id], sr=param.sr)
        audio_len = int(self.win_len * fs)

        if len(X) < audio_len:
            X = pad(X, audio_len)

        last_valid_start_sample = len(X) - audio_len
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        X_win = X[start_sample: start_sample + audio_len]
        X_win = Tensor(X_win)

        return X_win, y

    def __len__(self):
        return len(self.file_IDs)

class LoadTrainTripletData(Dataset):
    def __init__(self, file_IDs, speaker_IDs, labels, audio_path, win_len):
        self.file_IDs = file_IDs
        self.speaker_IDs = speaker_IDs
        self.labels = labels
        self.audio_path = audio_path
        self.win_len = win_len

    def __getitem__(self, index):

        if index % 2 == 0:
            X_id = random.choice(list({key: val for key, val in self.labels.items()
                                           if val == 0}))
            y = 0
            pos_id = random.choice(list({key: val for key, val in self.labels.items()
                                          if val == 0 and self.speaker_IDs[key]==self.speaker_IDs[X_id]}))
            neg_id = random.choice(list({key: val for key, val in self.labels.items()
                                          if val == 1 and self.speaker_IDs[key]==self.speaker_IDs[X_id]}))

        else:
            X_id = random.choice(list({key: val for key, val in self.labels.items()
                                           if val == 1}))
            y = 1
            pos_id = random.choice(list({key: val for key, val in self.labels.items()
                                          if val == 1 and self.speaker_IDs[key]==self.speaker_IDs[X_id]}))
            neg_id = random.choice(list({key: val for key, val in self.labels.items()
                                          if val == 0 and self.speaker_IDs[key]==self.speaker_IDs[X_id]}))



        X, fs = librosa.load(self.audio_path[X_id], sr=param.sr)
        pos, fs = librosa.load(self.audio_path[pos_id], sr=param.sr)
        neg, fs = librosa.load(self.audio_path[neg_id], sr=param.sr)

        audio_len = int(self.win_len * fs)

        last_valid_start_sample = len(X) - audio_len
        if not last_valid_start_sample <= 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        X_win = X[start_sample: start_sample + audio_len]
        X_win = Tensor(X_win)

        last_valid_start_sample = len(pos) - audio_len
        if not last_valid_start_sample <= 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        pos_pad = pad(pos, audio_len)
        pos_win = Tensor(pos_pad)

        last_valid_start_sample = len(neg) - audio_len
        if not last_valid_start_sample <= 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        neg_pad = pad(neg, audio_len)
        neg_win = Tensor(neg_pad)

        return X_win, y, pos_win, neg_win

    def __len__(self):
        return len(self.file_IDs)


class LoadEvalData(Dataset):
    def __init__(self, file_IDs, labels, audio_path, win_len):
        self.file_IDs = file_IDs
        self.labels = labels
        self.audio_path = audio_path
        self.win_len = win_len

    def __len__(self):
        return len(self.file_IDs)

    def __getitem__(self, index):
        key = self.file_IDs[index]
        X, fs = librosa.load(self.audio_path[key], sr=param.sr)

        audio_len = int(self.win_len * fs)

        if len(X) < audio_len:
            X = pad(X, audio_len)

        last_valid_start_sample = len(X) - audio_len
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        X_win = X[start_sample: start_sample + audio_len]
        X_win = Tensor(X_win)

        y = self.labels[key]
        return X_win, y, key


def get_text_from_voice(filename, large=False):

    wf = wave.open(filename, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        exit(1)

    if(large):
        model = Model(param.vosk_models_path + param.vosk_models[1])
    else:
        model = Model(param.vosk_models_path + param.vosk_models[0])

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

def checkCharacters(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base.en").to(device)
    special = []
    file = open(os.path.join(param.base_storage_path, 'badChars.txt'), 'w')

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

def generateCsv(df, final_path):

    df_vc = pd.DataFrame(columns=['speaker', 'filename', 'label', 'original_filename', 'audio_path', 'TTS', 'VC'])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(len(param.models_tts)):

            df_vc = df_vc.append(
                {'speaker': row['speaker'], 'filename': row["filename"] + "_TTS_" + str(i + 1),
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + ".wav"),
                 'TTS': str(i + 1),
                 'VC': '-'},
                ignore_index=True)

            df_vc = df_vc.append(
                {'speaker': row['speaker'], 'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_1", 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav"),
                 'TTS': str(i + 1),
                 'VC': '1'},
                ignore_index=True)

            df_vc = df_vc.append(
                {'speaker': row['speaker'], 'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_2", 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join(param.mod_eval_dir, row["filename"] + "_TTS_" + str(i + 1) + "_VC_2.wav"),
                 'TTS': str(i + 1),
                 'VC': '2'},
                ignore_index=True)

    df['TTS'] = '-'
    df['VC'] = '-'
    df_vc = pd.concat([df_vc, df], axis=0)
    df_vc = df_vc.sort_values('filename')
    df_vc.to_csv(final_path)

def generateCsvV2(df, final_path):

    df_vc = pd.DataFrame(columns=['filename', 'label', 'original_filename', 'audio_path'])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(len(param.models_tts)):

            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1),
                 'label': 'fake',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_ADD', row["filename"] + "_TTS_" + str(i + 1) + ".wav")},
                ignore_index=True)

            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_1", 'label': 'fake',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_ADD', row["filename"] + "_TTS_" + str(i + 1) + "_VC_1.wav")},
                ignore_index=True)

            df_vc = df_vc.append(
                {'filename': row["filename"] + "_TTS_" + str(i + 1) + "_VC_2", 'label': 'fake',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_ADD', row["filename"] + "_TTS_" + str(i + 1) + "_VC_2.wav")},
                ignore_index=True)

    df_vc = pd.concat([df_vc, df], axis=0)
    df_vc = df_vc.sort_values('filename')
    df_vc.to_csv(final_path)

def findNotSaved(final_path):

    df = pd.read_csv(final_path, sep=',', )
    counter = 0
    #for index, row in tqdm(df.iterrows(), total=len(df)):
#
 #       if not os.path.isfile(row['audio_path']):
  #          df.drop(index, inplace=True)
   #         counter = counter + 1

    df = df.sort_values('filename')
    df = df.reset_index()
    df.to_csv(final_path)
    print(str(counter))



def plot_roc_curve(labels, pred, legend=None):
    """
    Plot ROC curve.

    :param labels: groundtruth labels
    :type labels: list
    :param pred: predicted score
    :type pred: list
    :param legend: if True, add legend to the plot
    :type legend: bool
    :return:
    """
    # labels and pred bust be given in (N, ) shape

    def tpr5(y_true, y_pred):
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fp_sort = sorted(fpr)
        tp_sort = sorted(tpr)
        tpr_ind = [i for (i, val) in enumerate(fp_sort) if val >= 0.1][0]
        tpr01 = tp_sort[tpr_ind]
        return tpr01

    lw = 3

    fpr, tpr, _ = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    print('TPR5 = {:.3f}'.format(tpr5(labels, pred)))
    print('AUC = {:.3f}'.format(rocauc))
    print('EER = {:.3f}'.format(eer))
    print()
    if legend:
        plt.plot(fpr, tpr, lw=lw, label='$\mathrm{' + legend + ' - AUC = %0.2f}$' % rocauc)
    else:
        plt.plot(fpr, tpr, lw=lw, label='$\mathrm{AUC = %0.2f}$' % rocauc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel(r'$\mathrm{False\;Positive\;Rate}$', fontsize=18)
    plt.ylabel(r'$\mathrm{True\;Positive\;Rate}$', fontsize=18)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    # plt.show()


def extractirsweep(sweep_response, invsweepfft):
    """
    Extract the impulse response from a sweep signal.
    """

    if sweep_response.shape[0] > 1:
        sweep_response = sweep_response.T

    N = invsweepfft.shape[1]
    sweepfft = scipy.fft.fft(sweep_response, N)

    # convolve sweep with inverse sweep (freq domain multiply)

    ir = np.real(scipy.fft.ifft(invsweepfft * sweepfft))

    ir = np.roll(ir.T, ir.shape[1] // 2)

    irLin = ir[len(ir) // 2:]
    irNonLin = ir[:len(ir) // 2]

    return irLin, irNonLin


# function that saves the IR of the smartphones
def save_irs():

    mat = scipy.io.loadmat('/nas/public/exchange/dataset_smartphones/src/RIR_CODE/invsweepfft.mat')
    invsweepfft = mat['invsweepfft']

    write_path = '/nas/home/aorsatti/Pycharm/Tesi/data/IRs'

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    devices = ['huawei_nova_9', 'huawei_nova_9_se', 'huawei_p30', 'iphone_12_mini', 'iphone_13', 'iphone_xs_max', 'moto_edge_20', 'moto_edge_30', 'moto_g9_power', 'oneplus_nord', 'xiaomi_12', 'xiaomi_12_pro', 'xiaomi_12_x', 'realme_gt', 'redmagic']

    for idx, device in enumerate(devices):

        sweep_path = f'/nas/public/exchange/dataset_smartphones/edited/{device}/sweep.wav'
        sweep_recorded, fs = librosa.load(sweep_path, sr=None)

        irLin, irNonLin = extractirsweep(sweep_recorded, invsweepfft)

        # save the impulse responses as wav files in the data folder of this repository
        sf.write(os.path.join(write_path, f'ir_{device}.wav'), irLin[3000:6000,0],int(fs))

