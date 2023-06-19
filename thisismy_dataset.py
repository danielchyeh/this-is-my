import os
import json
import pandas as pd
import numpy as np
import argparse
from thisismy_utils import parse_dataset,load_thisismy,load_this_is_my_captions,load_this_is_my_distractors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default='this-is-my-dataset/', help='access data/label')
    parser.add_argument('--ANNO_DIR', type=str, default='this-is-my_test-set.json', help='access label file')
    parser.add_argument('--SEG_DIR', type=str, default='segments.csv', help='access csv file')
    parser.add_argument('--CAPT_DIR', type=str, default='this-is-my_eval-captions.csv', help='access caption file')
    args = parser.parse_args()

    ANNO_FILE = os.path.join(args.DATA_DIR, args.ANNO_DIR)
    SEGMENT_FILE = os.path.join(args.DATA_DIR, args.SEG_DIR)
    CAPTIONS_FILE = os.path.join(args.DATA_DIR, args.CAPT_DIR)

    train_x, train_y, eval_x, eval_y, train_class, eval_class, token2class, id2classname, token2item = load_thisismy(ANNO_FILE,SEGMENT_FILE)
    inst_ids, captions, gt, class_names = load_this_is_my_captions(CAPTIONS_FILE)
    distractor_segs = load_this_is_my_distractors(ANNO_FILE,SEGMENT_FILE)
