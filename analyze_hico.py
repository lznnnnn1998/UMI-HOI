import json
import matplotlib.pyplot as plt

anno_file = json.load(open("/home/Downloads/hico_20160224_det/annotations/test_hico.json", "r"))
# anno_file = json.load(open("/home/Downloads/hico_20160224_det/annotations/trainval_hico.json", "r"))
verb_number_unique = 0
verb_number_max = 0
verb_hist = [0 for _ in range(7)]
for anno in anno_file:
    verb_ids = []
    for hoi in anno['hoi_annotation']:
        if int(hoi['category_id']) not in verb_ids:
            verb_ids.append(int(hoi['category_id']))
    verb_number_unique += len(verb_ids)
    if len(verb_ids) >= verb_number_max:
        verb_number_max = len(verb_ids)
    verb_hist[len(verb_ids) - 1] += 1
print(f"total verb_number_unique={verb_number_unique}, \
      average verb_number_unique={verb_number_unique/len(anno_file)}, \
      verb_number_max={verb_number_max}")

plt.bar([i+1 for i in range(7)],  verb_hist)
# plt.savefig("/home/MasterThesis/pvic/analyze_hico_train.png")
plt.savefig("/home/MasterThesis/pvic/analyze_hico_test.png")
plt.close()