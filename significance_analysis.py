import sys
import math
# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu
import numpy as np

# https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
# https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/

# seed the random number generator
seed(1)
# generate two independent samples
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51

classmap = {0: "background",
            1: "building",
            2: "wood",
            3: "water",
            4: "road"}


def evaluate_wilcoxon(x, y, classid):
    class_name = classmap[classid]
   # print(data1)
    # compare samples
    stat, p = wilcoxon(x, y)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution for class %s (fail to reject H0)' % class_name)
    else:
        print('Different distribution for class %s (reject H0)' % class_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datax_file = sys.argv[1]
    if len(sys.argv) > 2:
        datay_file = sys.argv[2]
    if len(sys.argv) > 3:
        column = int(sys.argv[3])

    # extract data from x
    all_data_x = [[], [], [], [], []]
    all_data_y = [[], [], [], [], []]
    with open(datax_file, 'rt') as readx:
        lines = readx.readlines()
    for line in lines:
        if "iou" in line:
            split = line.split(",")
            unlabelled_iou = float(split[1]) if split[1] != "nan" else 0
            building_iou = float(split[2]) if split[2] != "nan" else 0
            wood_iou = float(split[3]) if split[3] != "nan" else 0
            water_iou = float(split[4]) if split[4] != "nan" else 0
            road_iou = float(split[5]) if split[5] != "nan" else 0
            all_data_x[0].append(unlabelled_iou)
            all_data_x[1].append(building_iou)
            all_data_x[2].append(wood_iou)
            all_data_x[3].append(water_iou)
            all_data_x[4].append(road_iou)

    # extract data from y
    with open(datay_file, 'rt') as ready:
        lines = ready.readlines()
    for line in lines:
        if "iou" in line:
            split = line.split(",")
            unlabelled_iou = float(split[1]) if split[1] != "nan" else 0
            building_iou = float(split[2]) if split[2] != "nan" else 0
            wood_iou = float(split[3]) if split[3] != "nan" else 0
            water_iou = float(split[4]) if split[4] != "nan" else 0
            road_iou = float(split[5]) if split[5] != "nan" else 0
            all_data_y[0].append(unlabelled_iou)
            all_data_y[1].append(building_iou)
            all_data_y[2].append(wood_iou)
            all_data_y[3].append(water_iou)
            all_data_y[4].append(road_iou)
    evaluate_wilcoxon(all_data_x[column][0:703], all_data_x[column][0:703], column)
