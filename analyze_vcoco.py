import json
import matplotlib.pyplot as plt
import torch
train_anno_file = json.load(open("/home/MasterThesis/pvic/vcoco/instances_vcoco_trainval.json", "r"))
test_anno_file = json.load(open("/home/MasterThesis/pvic/vcoco/instances_vcoco_test.json", "r"))
verb_number_unique = 0
verb_number_max = 0
verb_hist = [0 for _ in range(10)]
valid_samples = 0
for anno in test_anno_file['annotations']:
    
    if  len(anno['boxes_h']) == 0:
        continue
    valid_samples += 1
    verb_ids = torch.tensor(anno['actions']).unique().tolist()
    verb_number_unique += len(verb_ids)
    if len(verb_ids) >= verb_number_max:
        verb_number_max = len(verb_ids)
    verb_hist[len(verb_ids) - 1] += 1
print(f"total verb_number_unique={verb_number_unique}, \
      average verb_number_unique={verb_number_unique/valid_samples}, \
      verb_number_max={verb_number_max}")

plt.bar([i+1 for i in range(verb_number_max)],  verb_hist[:verb_number_max])
# plt.savefig("/home/MasterThesis/pvic/analyze_vcoco_train.png")
plt.savefig("/home/MasterThesis/pvic/analyze_vcoco_test.png")
plt.close()