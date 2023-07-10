import json
import pandas as pd
import numpy as np
import math

import h5py
from config import MSVDConfig

C=MSVDConfig


def load_metadata():
    df = pd.read_csv(C.caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    df = df.reset_index(drop=True)
    return df
# Taking only english language captions and removing some columns which we would not use in this notebook


def load_videos():
    f = h5py.File(C.video_fpath, 'r')
    return f


def load_splits():
    with open('/kaggle/input/video-data/video_caption_Train.list', 'r') as fin:
        train_vids = json.load(fin)
    with open('/kaggle/input/video-data/video_caption_valid.list', 'r') as fin:
        val_vids = json.load(fin)
    with open('/kaggle/input/video-data/video_caption_test.list', 'r') as fin:
        test_vids = json.load(fin)
    return train_vids, val_vids, test_vids


def save_video(fpath, vids, videos):
    fout = h5py.File(fpath, 'w')
    for vid in vids:
        fout[vid] = videos[vid][:]
    fout.close()
    print("Saved {}".format(fpath))


def save_metadata(fpath, vids, metadata_df):
    vid_indices = [ i for i, r in metadata_df.iterrows() if "{}_{}_{}".format(r[0], r[1], r[2]) in vids ]
    df = metadata_df.iloc[vid_indices]
# r[0] is video start. r[1] is start timer. r[2] is end timer 
# say -wa0umYJVGg_139_157 is in train/val/test list so 139 represents starting time. 157 ending and first part video id
    df.to_csv(fpath)
    print("Saved {}".format(fpath))


def split():
    videos = load_videos()
    metadata = load_metadata()

    train_vids, val_vids, test_vids = load_splits()

    save_video(C.train_video_fpath, train_vids, videos)
    save_video(C.val_video_fpath, val_vids, videos)
    save_video(C.test_video_fpath, test_vids, videos)

    save_metadata(C.train_metadata_fpath, train_vids, metadata)
    save_metadata(C.val_metadata_fpath, val_vids, metadata)
    save_metadata(C.test_metadata_fpath, test_vids, metadata)
    
    # make sure to create working directory to put these in kaggle
