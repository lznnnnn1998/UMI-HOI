# Docs

A variety of object detectors with different backbones are used for evaluation. For the HICO-DET dataset, to download the fine-tuned detector weights or attempt the fine-tuning yourself, refer to the instructions [here](https://github.com/fredzzhang/hicodet/tree/main/detections#detection-utilities).

## HICO-DET

### PViC-DETR-R50
```bash
# Training
DETR=base python main.py --pretrained checkpoints/detr-r50-hicodet.pth --output-dir outputs/pvic-detr-r50-hicodet --llava-answer-path text_folder --llava-token-path token_folder --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default --world-size 1 --batch-size 1 --epoch 15 --lr-drop 7
# Testing
DETR=base python main.py {same args as train...} --world-size 1 --batch-size 1 --eval --resume /path/to/model

# Caching detections for Matlab evaluation
DETR=base python main.py {same args as train...} --world-size 1 --batch-size 1 --cache --resume /path/to/model --output-dir matlab
```
### PViC-Defm-DETR-R50
```bash
# Training
DETR=advanced python main.py --pretrained checkpoints/defm-detr-r50-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/pvic-defm-detr-r50-hicodet --llava-answer-path text_folder --llava-token-path token_folder --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default --world-size 1 --batch-size 16 --epoch 15 --lr-drop 7

# Testing
DETR=advanced python main.py {same args as train...} --world-size 1 --batch-size 1 --eval --resume /path/to/model
# Caching detections for Matlab evaluation
DETR=base python main.py {same args as train...} --world-size 1 --batch-size 1 --cache --resume /path/to/model --output-dir matlab
```

### PViC-H-Defm-DETR-SwinL
```bash
# Training
DETR=advanced python main.py --backbone swin_large --use-checkpoint --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500 --pretrained checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/pvic-h-defm-detr-swinL-hicodet --llava-answer-path text_folder --llava-token-path token_folder --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default --world-size 1 --batch-size 1 --epoch 15 --lr-drop 7

# Testing
DETR=advanced python main.py {same args as train...} --world-size 1  --batch-size 1  --eval  --resume /path/to/model
# Caching detections for Matlab evaluation
DETR=advanced python main.py {same args as train...} --world-size 1  --batch-size 1   --cache  --resume /path/to/model  --output-dir matlab
```

## V-COCO

The fine-tuned detector weights can be downloaded below.

|Detector|DETR-R50|Defm-DETR-R50|H-Defm-DETR-R50|H-Defm-DETR-SwinL|
|:-|:-:|:-:|:-:|:-:|
|Weights|[159MB](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing)|[182MB](https://drive.google.com/file/d/1AR6IOotTC0BAkOikNMrIYQ5NubMTFtec/view?usp=sharing)|[183.4MB](https://drive.google.com/file/d/17MJK_uE5GJfZTn77Cc_LtgnOr8ULopcE/view?usp=sharing)|[833.1MB](https://drive.google.com/file/d/1GJs0BAJgJgFODE6oLxVE_N87nymXJLMy/view?usp=sharing)|

__IMPORTANT.__ There are still some issues with the V-COCO code after a codebase migration. These will be fixed soon.

### PViC-DETR-R50
```bash
# Training
DETR=base python main.py --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir outputs/pvic-detr-r50-vcoco --llava-answer-path text_folder --llava-token-path token_folder --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default --world-size 1 --batch-size 16 --epoch 15 --lr-drop 7

# Caching detections
DETR=base python main.py --dataset vcoco --data-root vcoco/ --partitions trainval test --world-size 1 --batch-size 1 --cache --resume /path/to/model --output-dir vcoco_cache {other args same as train...}
```


### PViC-H-Defm-DETR-SwinL
```bash
# Training
DETR=advanced python main.py --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone swin_large --use-checkpoint --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500  --pretrained checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-vcoco.pth  --output-dir outputs/pvic-h-defm-detr-swinL-vcoco --llava-answer-path text_folder --llava-token-path token_folder --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default --world-size 1 --batch-size 16 --epoch 15 --lr-drop 7
# Caching detections
DETR=advanced python main.py --dataset vcoco --data-root vcoco/ --partitions trainval test --world-size 1 --batch-size 1 --cache --resume /path/to/model --output-dir vcoco_cache {other args same as train...}
```
