import os
import math
import torch
import pocket
import pocket.advis
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import seaborn

from typing import Optional, List

from utils import DataFactory
from pvic_vis import build_detector
from configs import base_detector_args, advanced_detector_args
from action_list import action_list
warnings.filterwarnings("ignore")

@torch.no_grad()
def main(args):
    
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root, 
                          llava_answer_path=args.llava_answer_path, llava_token_path=args.llava_token_path)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_verbs = 117 if args.dataset == 'hicodet' else 24
    tp = 0
    fp = 0
    gt_sum = 0
    for i in range(args.example_num):
        image_name=dataset.dataset.filename(i)
        image, target, llava_answer, llava_feature = dataset[i]
        verb_ids_gt = dataset[i][1]['verb']
        llava_answer = torch.tensor(llava_answer).tolist()
        verb_ids_gt = torch.tensor(verb_ids_gt).unique().tolist()
        gt_sum += len(verb_ids_gt)
        for gt_id in verb_ids_gt:
            if gt_id in llava_answer:
                tp += 1
                llava_answer.remove(gt_id)
        fp += len(llava_answer)
        if i % 100 == 0:
            print(f"[{i}/{args.example_num}]")
    print(f"tp = {tp}, fp = {fp}, gt_sum = {gt_sum}")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(parents=[base_detector_args(),])
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--llava-answer-path', type=str)
    parser.add_argument('--llava-token-path', type=str)
    parser.add_argument('--example-num', default=1, type=int)
    parser.add_argument('--partition', type=str, default="test2015")
    args = parser.parse_args()
    main(args)