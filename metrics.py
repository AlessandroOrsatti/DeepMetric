import math
import numpy as np
import os
from scipy.spatial import distance
from tqdm import tqdm
import parameters as param
import pandas as pd

def addCosineDistance(df,emb_dir=param.embeddings_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        cosine_distance = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            cosine_distance = cosine_distance + distance.cosine(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
            df.loc[df['ID'] == row['ID'], "cosine distance"] = distance.cosine(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
        cosine_distance = cosine_distance/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "cosine distance"] = cosine_distance
    return df

def addEuclideanDistance(df, emb_dir=param.embeddings_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        euclidean_distance = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            euclidean_distance = euclidean_distance + distance.euclidean(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
            df.loc[df['ID'] == row['ID'], "euclidean distance"] = distance.euclidean(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
        euclidean_distance = euclidean_distance/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "euclidean distance"] = euclidean_distance
    return df

def addManhattanDistance(df, emb_dir=param.embeddings_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        manhattan_distance = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            manhattan_distance = manhattan_distance + distance.cityblock(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
            df.loc[df['ID'] == row['ID'], "manhattan distance"] = distance.cityblock(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
        manhattan_distance = manhattan_distance/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "manhattan distance"] = manhattan_distance
    return df

def addDotProduct(df, emb_dir=param.embeddings_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        dot_product = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            dot_product = dot_product + np.dot(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
            df.loc[df['ID'] == row['ID'], "dot product"] = np.dot(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
        dot_product = dot_product/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "dot product"] = dot_product
    return df

def addSquaredEuclideanDistance(df, emb_dir=param.embeddings_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        squared_euclidean_distance = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            squared_euclidean_distance = squared_euclidean_distance + distance.sqeuclidean(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
            df.loc[df['ID'] == row['ID'], "squared euclidean distance"] = distance.sqeuclidean(np.load(os.path.join(emb_dir, row['ID'] + '.npy')), np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy')))
        squared_euclidean_distance = squared_euclidean_distance/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "squared euclidean distance"] = squared_euclidean_distance
    return df

def addPredictionDistance(df):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    for original in tqdm(original_list, total=len(original_list)):
        prediction_distance = 0
        original_file = df.loc[df['ID']==original]
        for idx, row in df[df['original']==original].iterrows():
            prediction_distance = prediction_distance + math.sqrt(pow(row['prediction']-original_file['prediction'], 2))
            df.loc[df['ID'] == row['ID'], "prediction distance"] = math.sqrt(pow(row['prediction']-original_file['prediction'], 2))
        prediction_distance = prediction_distance/df[df['original']==original].shape[0]
        df.loc[df['ID'] == original, "prediction distance"] = prediction_distance
    return df

def addMetrics(df, emb_dir):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set


    for original in tqdm(original_list, total=len(original_list)):

        cosine_distance = 0
        euclidian_distance = 0
        manhattan = 0
        dot_product = 0
        squared_euclidean = 0

        original_file = df.loc[df['ID']==original]
        original_emb = np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy'))
        for idx, row in df[df['original']==original].iterrows():
            emb = np.load(os.path.join(emb_dir, row['ID'] + '.npy'))

            cosine_distance = cosine_distance + distance.cosine(emb, original_emb)
            euclidian_distance = euclidian_distance + distance.euclidean(emb, original_emb)
            manhattan = manhattan + distance.cityblock(emb, original_emb)
            dot_product = dot_product + np.dot(emb, original_emb)
            squared_euclidean = squared_euclidean + distance.sqeuclidean(emb, original_emb)

            df.loc[df['ID'] == row['ID'], "cosine distance"] = distance.cosine(emb, original_emb)
            df.loc[df['ID'] == row['ID'], "euclidean distance"] = distance.euclidean(emb, original_emb)
            df.loc[df['ID'] == row['ID'], "manhattan distance"] = distance.cityblock(emb, original_emb)
            df.loc[df['ID'] == row['ID'], "dot product"] = np.dot(emb, original_emb)
            df.loc[df['ID'] == row['ID'], "squared euclidean distance"] = distance.sqeuclidean(emb, original_emb)

        cosine_distance = cosine_distance/df[df['original']==original].shape[0]
        euclidian_distance = euclidian_distance / df[df['original'] == original].shape[0]
        manhattan = manhattan / df[df['original'] == original].shape[0]
        dot_product = dot_product / df[df['original'] == original].shape[0]
        squared_euclidean = squared_euclidean / df[df['original'] == original].shape[0]

        df.loc[df['ID'] == original, "cosine distance"] = cosine_distance
        df.loc[df['ID'] == original, "euclidean distance"] = euclidian_distance
        df.loc[df['ID'] == original, "manhattan distance"] = manhattan
        df.loc[df['ID'] == original, "dot product"] = dot_product
        df.loc[df['ID'] == original, "squared euclidean distance"] = squared_euclidean

    return df

def addMetricsConvolved(df, emb_dir):

    df_mod = pd.read_csv(os.path.join(param.csv_dir, 'asvspoof2019_eval_modified.txt'))

    original_list = df['ID'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    del original_set


    for original in tqdm(original_list, total=len(original_list)):

        cosine_distance = 0
        euclidian_distance = 0
        manhattan = 0
        dot_product = 0
        squared_euclidean = 0

        original_file = df.loc[df['ID']==original]
        original_id = original_file['convolved from'].iat[0]
        original_emb = np.load(os.path.join(emb_dir, original_file['ID'].iat[0] + '.npy'))
        for idx, row in df_mod[df_mod['original_filename']==original_id].iterrows():
            emb = np.load(os.path.join(param.embeddings_shortwinSmooth03_dir, row['filename'] + '.npy'))

            cosine_distance = cosine_distance + distance.cosine(emb, original_emb)
            euclidian_distance = euclidian_distance + distance.euclidean(emb, original_emb)
            manhattan = manhattan + distance.cityblock(emb, original_emb)
            dot_product = dot_product + np.dot(emb, original_emb)
            squared_euclidean = squared_euclidean + distance.sqeuclidean(emb, original_emb)

        cosine_distance = cosine_distance/df_mod[df_mod['original_filename']==original_id].shape[0]
        euclidian_distance = euclidian_distance / df_mod[df_mod['original_filename']==original_id].shape[0]
        manhattan = manhattan / df_mod[df_mod['original_filename']==original_id].shape[0]
        dot_product = dot_product / df_mod[df_mod['original_filename']==original_id].shape[0]
        squared_euclidean = squared_euclidean / df_mod[df_mod['original_filename']==original_id].shape[0]

        df.loc[df['ID'] == original, "cosine distance"] = cosine_distance
        df.loc[df['ID'] == original, "euclidean distance"] = euclidian_distance
        df.loc[df['ID'] == original, "manhattan distance"] = manhattan
        df.loc[df['ID'] == original, "dot product"] = dot_product
        df.loc[df['ID'] == original, "squared euclidean distance"] = squared_euclidean

    return df

def addMetricsV2(df):

    original_list = df['original'].values.tolist()
    original_set = set(original_list)
    original_list = list(original_set)
    original_list.remove('-')
    del original_set

    df = df[~df["ID"].str.contains('VC_2')]

    for original in tqdm(original_list, total=len(original_list)):

        cosine_distance = 0
        euclidian_distance = 0
        manhattan = 0
        dot_product = 0
        squared_euclidean = 0

        for idx, row in df[df['original']==original].iterrows():

            cosine_distance = cosine_distance + row['cosine distance']
            euclidian_distance = euclidian_distance + row['euclidean distance']
            manhattan = manhattan + row['manhattan distance']
            dot_product = dot_product + row['dot product']
            squared_euclidean = squared_euclidean + row['squared euclidean distance']

        cosine_distance = cosine_distance/df[df['original']==original].shape[0]
        euclidian_distance = euclidian_distance / df[df['original'] == original].shape[0]
        manhattan = manhattan / df[df['original'] == original].shape[0]
        dot_product = dot_product / df[df['original'] == original].shape[0]
        squared_euclidean = squared_euclidean / df[df['original'] == original].shape[0]

        df.loc[df['ID'] == original, "cosine distance (only VC 1)"] = cosine_distance
        df.loc[df['ID'] == original, "euclidean distance (only VC 1)"] = euclidian_distance
        df.loc[df['ID'] == original, "manhattan distance (only VC 1)"] = manhattan
        df.loc[df['ID'] == original, "dot product (only VC 1)"] = dot_product
        df.loc[df['ID'] == original, "squared euclidean distance (only VC 1)"] = squared_euclidean

    return df