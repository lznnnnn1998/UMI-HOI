"""
Visualise detected human-object interactions and
the cross-attention weights for SMHA, Register, self-attns.

Zhinan Li <zhinan.li@tum.de>

Technical University of Munich

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
import torch.nn.functional as F
from PIL import Image
from typing import Optional, List
from torchvision.transforms import ToPILImage, ToTensor
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
        image, image_name:str, output, attn_meta, action=None, 
        thresh=0.2, save_folder='./visualization/', 
        llava_answer=Optional[List[str]], verb_gt=Optional[List[str]],dataset=Optional[DataFactory]):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_folder+image_name.replace('.jpg','')):
        os.mkdir(save_folder+image_name.replace('.jpg', ''))
    print(f"processing:{image_name}")
    save_folder = save_folder.rstrip('/') + '/' + image_name.replace('.jpg', '') + '/'
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'].cpu() * scale_fct

    image_copy = image.copy()
    scores = output['scores']
    pred = output['labels']
    attn = attn_meta['attn_weights']
    cls_token_start = 0
    cls_token_end = attn_meta['num_query']
    backbone_token_start = cls_token_end
    backbone_token_end = cls_token_end + attn_meta['num_backbone_token']
    clip_token_start = backbone_token_end
    clip_token_end = backbone_token_end + attn_meta['num_clip_token'] - 1
    # that "1" is clip cls token
    answer_token_start = clip_token_end + 1
    answer_token_end = clip_token_end + 1 + attn_meta['num_answer_token']
    reg_token_start = answer_token_end
    
    # Visualise detected human-object pairs with attached scores
    attn_type = 'HO_backbone_attn'
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
        keep = torch.nonzero(scores >= thresh).squeeze(1).cpu()
        bx_h, bx_o = boxes[output['pairing'].cpu()].unbind(1)
        pocket.utils.draw_box_pairs(image, bx_h[keep].cpu(), bx_o[keep].cpu(), width=5)
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
        plt.close() 
        attn_type = "VLM_V_attn"
        for i in keep:
            ho_pair_idx = output["x"][i]
            attn_map = attn[:, ho_pair_idx, clip_token_start:clip_token_end].reshape(8, 24, 24) # resolution 384, patch_size = 16
            attn_image = image_copy.copy()
            attn_map_haed_std = attn_map.flatten(1).std(1)
            attn_map_th = attn_map_haed_std.sum() / 8
            attn_map_activated = (attn_map_haed_std >= attn_map_th).long()
            activated_ids = torch.where(attn_map_activated==1)[0].cpu()
            pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i], bx_o[i]]), width=4)
            _w, _h = attn_image.size
            attn_map = F.interpolate(attn_map.unsqueeze(1), size=(_h, _w), mode='bilinear').squeeze(1)
            # draw average attn
            pocket.advis.heatmap(attn_image, attn_map[activated_ids].mean(0, keepdim=True).cpu(), save_path=save_folder+f"pair_{i}_{attn_type}_avg_attn.png")
            plt.close()
            # draw each head's attn
            for j in range(8):
                if attn_map_activated[j] == 1:
                    flag = 'Act'
                else:
                    flag = 'nonAct'
                pocket.advis.heatmap(attn_image, attn_map[j: j+1].cpu(), save_path=save_folder+f"pair_{i}_{attn_type}_head_{j+1}_{flag}.png")
                plt.close()
        attn_type = "backbone_attn"
        for i in keep:
            ho_pair_idx = output["x"][i]
            attn_map = attn[:, ho_pair_idx, backbone_token_start:backbone_token_end]
            attn_map = attn_map.softmax(-1).reshape(8, math.ceil(h / 32), math.ceil(w / 32))
            contrast_factor = 4
            mean_ = attn_map.mean(dim=(-1, -2), keepdim=True)
            attn_map = contrast_factor * (attn_map - mean_) + mean_
            
            attn_image = image_copy.copy()
            attn_map_haed_std = attn_map.flatten(1).std(1)
            attn_map_th = attn_map_haed_std.sum() / 8
            attn_map_activated = (attn_map_haed_std >= attn_map_th).long()
            activated_ids = torch.where(attn_map_activated==1)[0].cpu()
            pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i], bx_o[i]]), width=4)
            _w, _h = attn_image.size
            attn_map = F.interpolate(attn_map.unsqueeze(1), size=(_h, _w), mode='bilinear').squeeze(1)
            # draw average attn
            pocket.advis.heatmap(attn_image, attn_map[activated_ids].mean(0, keepdim=True).cpu(), save_path=save_folder+f"pair_{i}_{attn_type}_avg_attn.png")
            plt.close()
            # draw each head's attn
            for j in range(8):
                if attn_map_activated[j] == 1:
                    flag = 'Act'
                else:
                    flag = 'nonAct'
                pocket.advis.heatmap(attn_image, attn_map[j: j+1].cpu(), save_path=save_folder+f"pair_{i}_{attn_type}_head_{j+1}_{flag}.png")
                plt.close()
        # draw text attention
        attn_type = 'VLM_T_attn'
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
            attn_map = attn[:, ho_pair_idx, answer_token_start:answer_token_end]
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
            seaborn.heatmap(attn_map.cpu(), xticklabels=llava_answer, yticklabels=flags)
            plt.title("GT:" + str(verb_gt).strip('[]') + '\n' + 
                      "Inference all: " + str(detect_results).strip('[]') + '\n' + 
                      "Inference this:" + detect_result)
            plt.xlabel("Verbs from VLM")
            plt.ylabel("Status of Heads")
            plt.tight_layout()
            plt.savefig(save_folder+f"pair_{i}_{attn_type}.png", bbox_inches='tight')
            plt.close()
        # draw reg attention
        attn_type = 'REG_attn'
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
            attn_map = attn[:, ho_pair_idx, reg_token_start:]
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
            seaborn.heatmap(attn_map.cpu(), xticklabels="auto", yticklabels=flags)
            plt.title(f"Attention to Registers\n" \
            f"Inference: {detect_result}")
            plt.xlabel("Register Indices")
            plt.ylabel("Status of Heads")
            plt.tight_layout()
            plt.savefig(save_folder+f"pair_{i}_{attn_type}.png", bbox_inches='tight')
            plt.close()
@torch.no_grad()
def main(args):
    
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root, 
                          llava_answer_path=args.llava_answer_path, llava_token_path=args.llava_token_path,
                          train_type=args.train_type)
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
    model.cuda()
    for _ in range(args.example_num):
        args.index += 1
        # vis ho query in matcher
        # x_queries = (before matcher, after matcher)
        h_queries = []
        o_queries = []
        hook1 = model.ho_matcher.h_mmf.register_forward_hook(
            lambda self, inputs, outputs: h_queries.append((inputs[0], outputs))
        )
        hook2 = model.ho_matcher.o_mmf.register_forward_hook(
            lambda self, inputs, outputs: o_queries.append((inputs[0], outputs))
        )
        dec_layer_idx = 0
        # vis backbone
        h_feature_pos = []
        o_feature_pos = []
        h_feature = []
        o_feature = []
        hook3 = model.decoder.layers[dec_layer_idx].backbone_pos_proj_h.register_forward_hook(
            lambda self, inputs, outputs: h_feature_pos.append(outputs)
        )
        hook4 = model.decoder.layers[dec_layer_idx].backbone_pos_proj_o.register_forward_hook(
            lambda self, inputs, outputs: o_feature_pos.append(outputs)
        )
        hook5 = model.decoder.layers[dec_layer_idx].h_feature_proj.register_forward_hook(
            lambda self, inputs, outputs: h_feature.append(outputs)
        )
        hook6 = model.decoder.layers[dec_layer_idx].o_feature_proj.register_forward_hook(
            lambda self, inputs, outputs: o_feature.append(outputs)
        )
        
        # vis answer token
        answer_token = []
        hook7 = model.decoder.layers[dec_layer_idx].answer_proj.register_forward_hook(
            lambda self, inputs, outputs: answer_token.append(outputs)
        )
        
        # vis attn
        attn_weights = []
        hook8 = model.decoder.layers[dec_layer_idx].self_attn.register_forward_hook(
            lambda self, inputs, outputs: attn_weights.append(outputs[1])
        )
        

        if args.image_path is None:
            image, target, llava_answer, llava_feature = dataset[args.index]
            image = image.to('cuda')
            llava_answer = torch.tensor(llava_answer).to('cuda')
            llava_feature = llava_feature.to('cuda')

            output = model(images=[image], targets=None, llava_answer=[llava_answer], llava_feature=[llava_feature])
            image = dataset.dataset.load_image(
                os.path.join(dataset.dataset._root,
                    dataset.dataset.filename(args.index)
            ))
        else:
            image = dataset.dataset.load_image(args.image_path)
            image_tensor, _ = dataset.transforms(image, None)
            output = model(images=[image_tensor], targets=None, llava_answer=[llava_answer], llava_feature=[llava_feature])
        

        ho_query_before = torch.cat((h_queries[0][0], o_queries[0][0]), dim=-1)
        ho_query_after = torch.cat((h_queries[0][1], o_queries[0][1]), dim=-1)
        del h_queries; del o_queries

        hw, bs, _ = h_feature[0].shape
        backbone_token = torch.cat((
            torch.cat((h_feature[0], h_feature_pos[0]), dim=-1).view(hw, bs, model.decoder.layers[0].h_dim), 
            torch.cat((o_feature[0], o_feature_pos[0]), dim=-1).view(hw, bs, model.decoder.layers[0].o_dim)
        ), dim=-1)
        del h_feature; del o_feature; del h_feature_pos; del o_feature_pos


        num_answer_token = len(answer_token[0])
        del answer_token

        num_reg = model.decoder.layers[-1].reg_num
        num_clip_token = attn_weights[0][0].shape[-1] - num_answer_token - len(ho_query_after) - hw - num_reg
        attn_meta = {
            'ho_query_before': ho_query_before,
            'ho_query_after': ho_query_after,
            'backbone_token': backbone_token.squeeze(1),
            'attn_weights': attn_weights[0][0],
            'num_query': len(ho_query_after),
            'num_backbone_token': hw,
            'num_answer_token': num_answer_token,
            'num_reg': num_reg,
            'num_clip_token':num_clip_token
        }

        
        hook1.remove(); hook2.remove(); hook3.remove(); hook4.remove(); 
        hook5.remove(); hook6.remove(); hook7.remove(); hook8.remove(); 
        verb_ids_gt = dataset[args.index][1]['verb']
        obj_ids_gt = dataset[args.index][1]['object']
        hoi_ids_gt = dataset[args.index][1]['hoi']
        verb_names_gt = []
        llava_answer_name = []
        for _ids in verb_ids_gt.unique():
            verb_names_gt.append(dataset.dataset._verbs[_ids])
        for _ids in llava_answer:
            llava_answer_name.append(dataset.dataset._verbs[_ids])
        
        visualise_entire_image(
            image=image, image_name=dataset.dataset.filename(args.index), output=output[0], attn_meta=attn_meta,
            action=args.action, thresh=args.action_score_thresh, 
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
    parser.add_argument('--train-type', default='default', type=str, choices=['default', 'RF_UC', 'NF_UC', 'UV', 'UO'])
    parser.add_argument('--sub-headnum', default=5, type=int, help='sub-headnum + obj-headnum = 8')
    parser.add_argument('--obj-headnum', default=3, type=int, help='sub-headnum + obj-headnum = 8')
    args = parser.parse_args()
    main(args)
