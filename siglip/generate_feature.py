from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import os
folder_dir = "/home/Downloads/hico_20160224_det/images/test2015/"
output_dir = "/home/MasterThesis/hico_siglip_feature/test/"
ckpt = "google/siglip2-giant-opt-patch16-384"
model = AutoModel.from_pretrained(ckpt, device_map="cuda:1").eval()
processor = AutoProcessor.from_pretrained(ckpt)
image_list = os.listdir(folder_dir)
total_num = len(image_list)
for idx, img_file in enumerate(image_list):
    print(f"processing: {img_file}, [{idx}/{total_num}]")
    image = Image.open(folder_dir + img_file).convert('RGB')
    file_name = img_file.split('.')[0] + '.pt'
    inputs = processor(images=image, return_tensors="pt")
    inputs.data['pixel_values'] = inputs.data['pixel_values'].to('cuda:1')
    with torch.no_grad():
        last_hs, pooled_output = model.get_image_features(**inputs)
        last_hs = last_hs.squeeze(0)
        feature = torch.cat([last_hs, pooled_output], dim=0)
        torch.save(feature.cpu(), output_dir + file_name)