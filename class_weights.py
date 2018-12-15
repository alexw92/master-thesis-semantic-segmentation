import sys, os
import glob
import numpy as np
import skimage.io as io
from tqdm import trange

np.set_printoptions(precision=15)

files = glob.glob(os.path.join(sys.argv[1], "*.png"))
nfiles = len(files)

lbl_counts = {}
# kaggle label counts
# lbl_counts = {0: 0.615830, 1: 0.038690, 2: 0.004052, 3: 0.008817,
# 			  4: 0.029346, 5: 0.108000, 6: 0.189018, 7: 0.005066,
# 			  8: 0.001049, 9: 1.176852e-05, 10: 0.000120}

# vaihingen weights
# {0: 0.27496586356764924, 1: 0.2600085789871505, 2: 0.20469702380952381,
#  3: 0.23992597316704442, 4: 0.012058994708994705, 5: 0.008343565759637187}

#lbl_counts = {0: 0.1515, 1: 0.1441, 2: 0.2456, 3: 0.02239, 4: 0.1113, 5: 0.3250}
# lbl_counts = {0: 0.27496586356764924, 1: 0.2600085789871505, 2: 0.20469702380952381,
#               3: 0.23992597316704442, 4: 0.012058994708994705, 5: 0.008343565759637187}
if not lbl_counts:
    for i in trange(nfiles, leave=True):
        f = files[i]
        img = io.imread(f)[:,:] # first channel of gray label image
        id, counts = np.unique(img, return_counts=True)
        # normalize on image
        counts = counts / float(sum(counts))
        for i in range(len(id)):
            if id[i] in lbl_counts.keys():
                lbl_counts[id[i]] += counts[i]
            else:
                lbl_counts[id[i]] = counts[i]

    # normalize on training set
    for k in lbl_counts:
        lbl_counts[k] /= nfiles

print("##########################")
print("class probability:")
for k in lbl_counts:
    print("%i: %f" % (k, lbl_counts[k]))
print("##########################")

# normalize on median freuqncy
med_frequ = np.median(list(lbl_counts.values()))
lbl_weights = {}
for k in lbl_counts:
    lbl_weights[k] = med_frequ / lbl_counts[k]

print("##########################")
print("median frequency balancing:")
for k in lbl_counts:
    print("%i: %f" % (k, lbl_weights[k]))
print("##########################")

# class weight for classes that are not present in labeled image
missing_class_weight = 100000

max_class_id = np.max(list(lbl_weights.keys()))+1

# printformated output for caffe prototxt
print("########################################################")
print("### caffe SoftmaxWithLoss format #######################")
print("########################################################")
print("  loss_param: {\n"\
"    weight_by_label_freqs: true")
#"\n    ignore_label: 0"
for k in range(max_class_id):
    if k in lbl_weights:
        print("    class_weighting: "+ str(lbl_weights[k]))
    else:
        print("    class_weighting: "+ str(missing_class_weight))
print("  }")
print("########################################################")