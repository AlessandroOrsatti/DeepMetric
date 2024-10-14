
import numpy as np
import torch
from utils import generationDataFrameV2
import pandas as pd
import os
from tqdm import tqdm
import parameters as param

#get device and paths
device = "cuda" if torch.cuda.is_available() else "cpu"

#generate dataset and speaker list
df = generationDataFrameV2()

df_mod = pd.DataFrame(columns=['filename', 'label', 'original_filename', 'audio_path'])

        # for loop to pass every file path in temp
for _, row in tqdm(df.iterrows(), total=len(df))  :


            df_mod = df_mod.append(
                    {'filename': row["filename"] + "_TTS_1",
                     'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2', row["filename"] + "_TTS_1.wav")},
                    ignore_index=True)

            df_mod = df_mod.append(
                    {'filename': row["filename"] + "_TTS_2",
                     'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                row["filename"] + "_TTS_2.wav")},
                    ignore_index=True)

            df_mod = df_mod.append(
                    {'filename': row["filename"] + "_TTS_3",
                     'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                row["filename"] + "_TTS_3.wav")},
                    ignore_index=True)

            df_mod = df_mod.append(
                    {'filename': row["filename"] + "_TTS_4",
                     'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                row["filename"] + "_TTS_4.wav")},
                    ignore_index=True)

            df_mod = df_mod.append(
                    {'filename': row["filename"] + "_TTS_5",
                     'label': 'spoof',
                     'original_filename': row["filename"],
                     'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                                row["filename"] + "_TTS_5.wav")},
                    ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_1_VC_1",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_1_VC_1.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_2_VC_1",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_2_VC_1.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_3_VC_1",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_3_VC_1.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_4_VC_1",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_4_VC_1.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_5_VC_1",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_5_VC_1.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_1_VC_2",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_1_VC_2.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_2_VC_2",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_2_VC_2.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_3_VC_2",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_3_VC_2.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_4_VC_2",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_4_VC_2.wav")},
                ignore_index=True)

            df_mod = df_mod.append(
                {'filename': row["filename"] + "_TTS_5_VC_2",
                 'label': 'spoof',
                 'original_filename': row["filename"],
                 'audio_path': os.path.join('/nas/home/aorsatti/Pycharm/Tesi/data/modified_dataset2',
                                            row["filename"] + "_TTS_5_VC_2.wav")},
                ignore_index=True)


df_mod = pd.concat([df_mod, df], axis=0)
df_mod = df_mod.sort_values('filename')
df_mod.to_csv('/nas/home/aorsatti/Pycharm/Tesi/data/csv/modified_dataset2.csv')