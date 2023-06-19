import os
import json
import pandas as pd
import numpy as np


# DATA_DIR = 'this-is-my-dataset/'
# ANNO_FILE = os.path.join(DATA_DIR, 'this-is-my_test-set.json')
# SEGMENT_FILE = os.path.join(DATA_DIR, 'segments.csv')

#CAPTIONS_FILE = os.path.join(DATA_DIR, 'this-is-my_eval-captions.csv')


def parse_dataset(annofile, segments_file, load_distractors=False):
    with open(annofile, "r") as f:
        anchors2segs = json.load(f)

    df = pd.read_csv(segments_file, header=0, delimiter=',')
    df_anchors = df[df['is_anchor']]
    anchor_segs = df_anchors.iloc[:, 0].to_list()
    segs = []
    inst_ids = []
    id = 0
    token2class = {}
    token2item = {}
    for i, s in enumerate(anchor_segs):
        if s in anchors2segs.keys():
            token2class[id] = df_anchors.iloc[i]['query']
            token2item[id] = df_anchors.iloc[i]['instance_name']
            segs.append(s)
            segs += anchors2segs[s]
            inst_ids += [id]*(len(anchors2segs[s])+1)
            id += 1
        else:
            print('Missing annotation for: {}'.format(s))
    print('Found {} segments across {} instances'.format(len(segs), id))
    assert len(segs) == len(inst_ids)

    vid_ids = [df[df['segment_id'] == s]['video_id'].values[0] for s in segs]

    if load_distractors:
        other_segs = df[df['segment_id'].isin(segs) == False]
        other_segs_same_vids = other_segs[other_segs['video_id'].isin(vid_ids)]
        distractor_segs = other_segs_same_vids['segment_id'].to_list()
        print('Found {} distractor segments'.format(len(distractor_segs)))
        return distractor_segs
    else:
        return np.array(segs), np.array(inst_ids), np.array(vid_ids), token2class, token2item


def load_thisismy(ANNO_FILE, SEGMENT_FILE):
    print('=========ThisIsMy DATASET============')
    segs, ids, vid_ids, token2class, token2item = parse_dataset(ANNO_FILE, SEGMENT_FILE)

    unique_ids = np.unique(ids)

    first_idx_per_inst = np.stack([np.argmax(ids==i) for i in unique_ids])
    train_vid_ids = vid_ids[first_idx_per_inst]
    train_idx = np.squeeze(np.concatenate([np.argwhere(vid_ids==i) for i in train_vid_ids]))
    test_idx = np.array([i for i in range(len(segs)) if i not in train_idx])

    train_vids = np.unique(vid_ids[train_idx])
    print('Training videos: {}'.format(len(train_vids)))

    train_x = segs[train_idx]
    train_y = ids[train_idx]

    eval_x = segs[test_idx]
    eval_y = ids[test_idx]

    train_class = np.array([token2class[t] for t in train_y])
    eval_class = np.array([token2class[t] for t in eval_y])

    classes = sorted(np.unique(train_class))
    class2label = {c: i for i, c in enumerate(classes)}
    id2classname = {i: c for i, c in enumerate(classes)}
    token2class = {k: class2label[v] for k, v in token2class.items()}

    train_class = np.array([class2label[t] for t in train_class])
    eval_class = np.array([class2label[t] for t in eval_class])

    print('Number of train Concepts: {}'.format(len(np.unique(train_y))))
    print('Number of eval Concepts: {}'.format(len(np.unique(eval_y))))

    print('Number of train segments: {}'.format(len(train_x)))
    print('Number of eval segments: {}'.format(len(eval_x)))

    return train_x, train_y, eval_x, eval_y, train_class, eval_class, token2class, id2classname, token2item


def load_this_is_my_captions(cap_file):
    df = pd.read_csv(cap_file, header=0, delimiter=',')
    captioned = df[~df['caption'].isna()]
    gt = np.array(captioned.index).reshape((-1, 1))
    captions = captioned['caption'].to_list()
    inst_ids = captioned['ids'].to_list()
    class_names = captioned['class'].to_list()
    return inst_ids, captions, gt, class_names


def load_this_is_my_distractors(ANNO_FILE,SEGMENT_FILE):
    segs = parse_dataset(ANNO_FILE, SEGMENT_FILE, load_distractors=True)
    segs = np.array(segs)
    return segs

