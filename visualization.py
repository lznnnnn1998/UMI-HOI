"""
Visualise detected human-object interactions and
the cross-attention weights.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

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

def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(
        image, image_name:str, output, attn, action=None, 
        thresh=0.2, attn_type='qk_attn', save_folder='./visualization/', 
        llava_answer=Optional[List[str]], verb_gt=Optional[List[str]],dataset=Optional[DataFactory]):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_folder+image_name.replace('.jpg','')):
        os.mkdir(save_folder+image_name.replace('.jpg', ''))
    print(f"processing:{image_name}, attn_type:{attn_type}")
    save_folder = save_folder.rstrip('/') + '/' + image_name.replace('.jpg', '') + '/'
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct

    image_copy = image.copy()
    scores = output['scores']
    pred = output['labels']
    # Visualise detected human-object pairs with attached scores
    if attn_type=='qk_attn':
        if action is not None:
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
            bx_h, bx_o = boxes[output['pairing']].unbind(1)
            pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
            plt.imshow(image)
            plt.axis('off')

            for i in range(len(keep)):
                detect_result = dataset.dataset._verbs[output["labels"][keep[i]]]
                txt = plt.text(*bx_h[keep[i], :2], f"{detect_result}:{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
                txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
                plt.draw()
            
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_folder+image_name.replace('jpg', 'png'), bbox_inches="tight", pad_inches=0)

            for i in keep:
                ho_pair_idx = output["x"][i]
                attn_map = attn[0, :, ho_pair_idx].reshape(8, math.ceil(h / 32), math.ceil(w / 32))
                attn_map_haed_std = attn_map.flatten(1).std(1)
                attn_map_th = attn_map_haed_std.sum() / 8
                attn_map_activated = (attn_map_haed_std >= attn_map_th).long()

                attn_image = image_copy.copy()
                pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i], bx_o[i]]), width=4)
                if args.avg_attn:
                    pocket.advis.heatmap(attn_image, attn_map.mean(0, keepdim=True), save_path=save_folder+f"pair_{i}_{attn_type}_avg_attn.png")
                    plt.close()
                else:
                    for j in range(8):
                        if attn_map_activated[j] == 1:
                            flag = 'Act'
                        else:
                            flag = 'nonAct'
                        pocket.advis.heatmap(attn_image, attn_map[j: j+1], save_path=save_folder+f"pair_{i}_{attn_type}_head_{j+1}_{flag}.png")
                        plt.close()
        elif action is None:
            keep = torch.nonzero(scores >= thresh).squeeze(1)
            bx_h, bx_o = boxes[output['pairing']].unbind(1)
            pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
            plt.imshow(image)
            plt.axis('off')
            text_positions = torch.zeros((1, 0, 2), device=bx_h.device)
            
            for i in range(len(keep)):
                text_position = bx_h[keep[i], :2]
                text_gap = 30
                distance = ((text_positions - text_position.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
                if i == 0:
                    text_positions = torch.concat((text_positions, text_position.unsqueeze(0).unsqueeze(0)), dim=1)
                elif distance.min() > text_gap ** 2:
                    text_positions = torch.concat((text_positions, text_position.unsqueeze(0).unsqueeze(0)), dim=1)
                else:
                    while distance.min() <= text_gap ** 2:
                        last_text_position = text_positions[0][torch.argmin(distance, 1)]
                        text_position = torch.zeros_like(bx_h[keep[i], :2])
                        text_position[0] = last_text_position.flatten()[0]
                        text_position[1] = last_text_position.flatten()[1] + text_gap + 1
                        distance = ((text_positions - text_position.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
                    text_positions = torch.concat((text_positions, text_position.unsqueeze(0).unsqueeze(0)), dim=1)
                detect_result = dataset.dataset._verbs[output["labels"][keep[i]]]
                txt = plt.text(*text_position.flatten(), f"{detect_result}:{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
                txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
                plt.draw()
            
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_folder+image_name.replace('jpg', 'png'), bbox_inches="tight", pad_inches=0)

            for i in keep:
                ho_pair_idx = output["x"][i]
                attn_map = attn[0, :, ho_pair_idx].reshape(8, math.ceil(h / 32), math.ceil(w / 32))
                attn_image = image_copy.copy()
                attn_map_haed_std = attn_map.flatten(1).std(1)
                attn_map_th = attn_map_haed_std.sum() / 8
                attn_map_activated = (attn_map_haed_std >= attn_map_th).long()
                activated_ids = torch.where(attn_map_activated==1)[0]
                pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i], bx_o[i]]), width=4)
                if args.avg_attn:
                    pocket.advis.heatmap(attn_image, attn_map[activated_ids].mean(0, keepdim=True), save_path=save_folder+f"pair_{i}_{attn_type}_avg_attn.png")
                    plt.close()
                else:
                    for j in range(8):
                        if attn_map_activated[j] == 1:
                            flag = 'Act'
                        else:
                            flag = 'nonAct'
                        pocket.advis.heatmap(attn_image, attn_map[j: j+1], save_path=save_folder+f"pair_{i}_{attn_type}_head_{j+1}_{flag}.png")
                        plt.close() 
    elif attn_type=='feature_cross_attn':
        attn_map = attn[0].max(2)[0].reshape(8, math.ceil(h / 32), math.ceil(w / 32))
        attn_map = attn_map ** 2
        attn_map_haed_std = attn_map.flatten(1).std(1)
        attn_map_th = attn_map_haed_std.sum() / 8
        attn_map_activated = (attn_map_haed_std >= attn_map_th).long()
        attn_image = image_copy.copy()
        if args.avg_attn:
            pocket.advis.heatmap(attn_image, attn_map.mean(0, keepdim=True), save_path=save_folder+f"{attn_type}_avg_attn.png")
            plt.close()
        else:
            for j in range(8):
                if attn_map_activated[j] == 1:
                    flag = 'Act'
                else:
                    flag = 'nonAct'
                pocket.advis.heatmap(attn_image, attn_map[j: j+1], save_path=save_folder+f"{attn_type}_head_{j+1}_{flag}.png")
                plt.close() 
        pass
    elif attn_type == 'cross_attn_L1' or attn_type == 'cross_attn_L2':
        keep = torch.nonzero(scores >= thresh).squeeze(1)
        detect_results_ids = []
        detect_results = []
        for i in keep:
            if output["labels"][i] not in detect_results_ids:
                detect_results_ids.append(output["labels"][i])
        for _ids in detect_results_ids:
            detect_results.append(dataset.dataset._verbs[_ids])
        for i in keep:
            ho_pair_idx = output["x"][i]
            detect_result = dataset.dataset._verbs[output["labels"][i]]
            attn_map = attn[0, :, ho_pair_idx]
            attn_map_haed_std = attn_map.flatten(1).std(1)
            attn_map_th = attn_map_haed_std.sum() / 8
            attn_map_activated = (attn_map_haed_std >= attn_map_th).long()
            flags = []
            for j in range(8):
                if attn_map_activated[j] == 1:
                    flags.append('Activated')
                else:
                    flags.append('Non-activated')
            plt.clf()
            seaborn.heatmap(attn_map, xticklabels=llava_answer, yticklabels=flags)
            plt.title("GT:" + str(verb_gt).strip('[]') + '\n' + 
                      "Inference all: " + str(detect_results).strip('[]') + '\n' + 
                      "Inference this:" + detect_result)
            plt.tight_layout()
            plt.savefig(save_folder+f"pair_{i}_{attn_type}.png", bbox_inches='tight')
            plt.close()
        pass
@torch.no_grad()
def main(args):
    
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root, 
                          llava_answer_path=args.llava_answer_path, llava_token_path=args.llava_token_path)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_verbs = 117 if args.dataset == 'hicodet' else 24

    model = build_detector(args, conversion)
    model.eval()
    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")
    for _ in range(args.example_num):
        args.index += 1
        attn_weights = {
            'qk_attn': [],
            'feature_cross_attn':[],
            'cross_attn_L2':[],
            'cross_attn_L1':[]
        }
        hook1 = model.decoder.layers[-1].qk_attn.register_forward_hook(
            lambda self, input, output: attn_weights['qk_attn'].append(output[1])
        )
        hook2 = model.decoder.layers[-1].feature_cross_attn.register_forward_hook(
            lambda self, input, output: attn_weights['feature_cross_attn'].append(output[1])
        )
        hook3 = model.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: attn_weights['cross_attn_L2'].append(output[1])
        )
        hook4 = model.decoder.layers[-2].cross_attn.register_forward_hook(
            lambda self, input, output: attn_weights['cross_attn_L1'].append(output[1])
        )
        

        if args.image_path is None:
            image, target, llava_answer, llava_feature = dataset[args.index]
            output = model(images=[image], targets=None, llava_answer=[llava_answer], llava_feature=[llava_feature])
            image = dataset.dataset.load_image(
                os.path.join(dataset.dataset._root,
                    dataset.dataset.filename(args.index)
            ))
        else:
            image = dataset.dataset.load_image(args.image_path)
            image_tensor, _ = dataset.transforms(image, None)
            output = model(images=[image_tensor], targets=None, llava_answer=[llava_answer], llava_feature=[llava_feature])

        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()
        verb_ids_gt = dataset[args.index][1]['verb']
        obj_ids_gt = dataset[args.index][1]['object']
        hoi_ids_gt = dataset[args.index][1]['hoi']
        verb_names_gt = []
        llava_answer_name = []
        for _ids in verb_ids_gt.unique():
            verb_names_gt.append(dataset.dataset._verbs[_ids])
        for _ids in llava_answer:
            llava_answer_name.append(dataset.dataset._verbs[_ids])
        for k, v in attn_weights.items():
            visualise_entire_image(
                image=image, image_name=dataset.dataset.filename(args.index), output=output[0], attn=v[0],
                action=args.action, thresh=args.action_score_thresh, attn_type=k, 
                llava_answer=llava_answer_name, verb_gt=verb_names_gt, dataset=dataset
            )
        
if __name__ == "__main__":
    
    if "DETR" not in os.environ:
        raise KeyError(f"Specify the detector type with env. variable \"DETR\".")
    elif os.environ["DETR"] == "base":
        parser = argparse.ArgumentParser(parents=[base_detector_args(),])
        parser.add_argument('--detector', default='base', type=str)
        parser.add_argument('--raw-lambda', default=2.8, type=float)
    elif os.environ["DETR"] == "advanced":
        parser = argparse.ArgumentParser(parents=[advanced_detector_args(),])
        parser.add_argument('--detector', default='advanced', type=str)
        parser.add_argument('--raw-lambda', default=1.7, type=float)

    parser.add_argument('--partition', type=str, default="test2015")

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--avg-attn', action='store_true', default=False)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
        help="Path to an image file.")
    parser.add_argument('--llava-answer-path', type=str)
    parser.add_argument('--llava-token-path', type=str)
    parser.add_argument('--example-num', default=1, type=int)
    args = parser.parse_args()
    main(args)
