import glob
import os
from PIL import Image
import sys
from tqdm import trange
import numpy as np

class_name = [
    'Unlabelled',
    'Building',
    'Wood',
    'Water',
    'Road',
    'Residential'
]

kaggle_class_names = [
    'OTHER', 'BUILDING', 'MIS_STRUCTURES', 'ROAD', 'TRACK',
    'TREES', 'CROPS', 'WATERWAY', 'STANDING_WATER',
    'VEHICLE_LARGE', 'VEHICLE_SMALL'
]

vaihingen_class_names = ['Impervious_surfaces',
'Buildings', 'Low vegetation', 'Tree',
'Car', 'Clutter']


def fast_hist(gt, pred, n_clss):
    # true false mask where gt is valid
    k = (gt >= 0) & (gt < n_clss)
    return np.bincount(n_clss * gt[k].astype(int) + pred[k], minlength=n_clss ** 2).reshape(n_clss, n_clss)


def get_hist(predictions, labels, c_clss):
    num_class = c_clss
    batch_size = 1#predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        lhist = fast_hist(labels[i].flatten(), predictions[i].flatten(), num_class)
        hist += lhist
    return hist


def print_hist_summery(hist, dataset=None):
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    num_class = hist.shape[0]
    if num_class==11:
        print("class shape last hist summary "+str(num_class))
        cl_name = kaggle_class_names
    elif dataset== 'vaihingen':
        cl_name = vaihingen_class_names
    elif 'nores' in dataset:
        cl_name = class_name
    else:
        cl_name = class_name
    for cls in range(hist.shape[0]):
        iou = iu[cls]
        if float(hist.sum(1)[cls]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
        print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12), acc, iou))


def main():
    path = sys.argv[1] # './segmentation_results'
   # path = './segmentation_results'
    l_list = []
    p_list = []
    os.chdir(path)
    for file in glob.glob('*_nc_label.png'):
        l_list.append(file)
    for file in glob.glob('*_nc_prediction.png'):
        p_list.append(file)
    p_list = sorted(p_list)
    l_list = sorted(l_list)
    dataset_name = sys.argv[2]  # './segmentation_results' 'de_top15_nores' 20
    num_images = int(sys.argv[3])
    num_class = int(sys.argv[4])

    # get class names
    if 'vaihingen' in dataset_name:
        cl_name = vaihingen_class_names
    elif 'kaggle' in dataset_name:
        cl_name = kaggle_class_names
    else:
        cl_name = class_name

    hist = np.zeros((num_class, num_class))
    eval_file = open(dataset_name, 'wt+')
    for i in trange(num_images):
        pred = p_list[i]
        gt = l_list[i]
        print(pred)
        print(gt)
        pred = np.asarray(Image.open(pred))
        label = np.asarray(Image.open(gt))
        lhist = get_hist(predictions=pred, labels=label, c_clss=num_class)

        lacc = np.diag(lhist).sum() / lhist.sum()
        liu = np.diag(lhist) / (lhist.sum(1) + lhist.sum(0) - np.diag(lhist))
        acc_line = "acc"
        iou_line = "iou"
        for cls in range(lhist.shape[0]):
            iou = liu[cls]
            if float(lhist.sum(1)[cls]) == 0:
                local_acc = 0.0
            else:
                local_acc = np.diag(lhist)[cls] / float(lhist.sum(1)[cls])
            acc_line = acc_line + ", " + str(local_acc)
            iou_line = iou_line + ", " + str(iou)

        eval_file.write(acc_line + "\n")
        eval_file.write(iou_line + "\n")
        eval_file.flush()

        hist += lhist
    eval_file.close()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('number classes: '+str(hist.shape[0]))
    for cls in range(hist.shape[0]):
        iou = iu[cls]
        if float(hist.sum(1)[cls]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
        print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12), acc, iou))



    # hist = get_hist(predictions=preds, labels=labels)
    # print_hist_summery(hist=hist, dataset=dataset_name)




if __name__ == "__main__":
    main()
