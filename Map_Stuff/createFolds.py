
import glob
import sys
import os
import re
import random as ran
import numpy as np
from PIL import Image
from tqdm import trange


def createFoldsByCities(datadir, folds, tries=1000):
    os.chdir(datadir)
    l = dict()
    for f in glob.glob("*.png"):
        pre = f[:4]
        if pre in l:
            l[pre] = l[pre]+1
        else:
            l[pre] = 1

    num_cities = len(l)
    if (num_cities % folds) != 0:
        print("Invalid number of folds %d for %d cities!" % (folds, num_cities))
        return
    print("Dividing %d cities into %d folds..." % (num_cities, folds))
    # balance to folds with lowest variance considering number
    cities_per_fold = num_cities//folds
    folds_best = None
    var_min = 10000
    for t in range(tries):
        cities = list(l.keys())
        ran.shuffle(cities)
        folds_dict = dict()
        folds_num_dict = dict()

        # create fold
        for fold in range(folds):
            fold_list = []
            for i in range(cities_per_fold):
                city = cities.pop()
                fold_list.append(city)
                if fold not in folds_num_dict:
                    folds_num_dict[fold] = 0
                folds_num_dict[fold] = folds_num_dict[fold]+l[city]
            folds_dict[fold] = fold_list

        # eval fold
        fold_distrib = [(folds_num_dict[i]) for i in list(folds_num_dict.keys())]

        var_loc = np.std(fold_distrib)
        if var_loc < var_min:
            var_min = var_loc
            folds_best = folds_dict
            print(fold_distrib)
    print("Best fold splitting with std %d" % var_min)
    print(folds_best)


def evaluate_feature_distrib_from_GT(datadir="../ANN_Data/de_top14_cropped",
                                     img_list="../Segmentation_Models/ICNet/list/de_top14_train_list.txt"):

    feature_ids = {
        "unlabelled": 0,
        "building": 1,
        "wood": 2,
        "water": 3,
        "road": 4,
        "residential": 5
    }
    # load images
    if img_list is None:
        datadir = datadir+"/gt_data"
        masks = []
        old_dir = os.getcwd()
        os.chdir(datadir)
        for mask in glob.glob("*.png"):
            mask = os.path.join(datadir, mask)
            masks.append(mask)
        os.chdir(old_dir)
    else:
        f = open(img_list, 'r')
        masks = []
        for line in f:
            try:
                image, mask = line[:-1].split(' ')
            except ValueError:  # Adhoc for test.
                image = mask = line.strip("\n")
            mask = os.path.join(datadir, mask)
            image = os.path.join(datadir, image)
            mask = mask.strip()
            if not os.path.isfile(mask):
                raise ValueError('Failed to find file: ' + mask)
            if not os.path.isfile(image):
                continue

            masks.append(mask)

    total_feature_distrib = {
        "wood": 0,
        "building": 0,
        "water": 0,
        "road": 0,
        "unlabelled": 0,
        "residential": 0
    }
    for idx in trange(len(masks), desc='Evaluate_Feature_Distribution_From_GT', leave=True):
        im_filename = masks[idx]
        im = np.asarray(Image.open(im_filename))
        # add to sum
        distrib = evaluate_feature_distrib(im)

        for feature_class in total_feature_distrib.keys():
            total_feature_distrib[feature_class] += distrib[feature_ids[feature_class]]

    # calc average
    for feature_class in total_feature_distrib.keys():
        total_feature_distrib[feature_class] /= len(masks)

    return total_feature_distrib


def evaluate_feature_distrib_for_city(gt_datadir="../ANN_Data/de_top15_cropped/gt_data"):
    feature_ids = {
        "unlabelled": 0,
        "building": 1,
        "wood": 2,
        "water": 3,
        "road": 4,
        "residential": 5
    }
    # load images
    datadir = gt_datadir
    masks = []
    old_dir = os.getcwd()
    os.chdir(datadir)
    for mask in glob.glob("*.png"):
        mask = os.path.join(datadir, mask)
        masks.append(mask)
    os.chdir(old_dir)
    print(os.path.basename(masks[0]))
    cities = dict()
    for f in masks:
        fname = os.path.basename(f)
        filename_regex = "GT_([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
        m = re.search(filename_regex, fname)
        city = m.group(1)
        if city not in cities:
            cities[city] = []
        else:
            cities[city].append(f)

    city_distribs = dict()
    for k in cities.keys():
        city_images = cities[k]
        total_feature_distrib = {
            "wood": 0,
            "building": 0,
            "water": 0,
            "road": 0,
            "unlabelled": 0,
            "residential": 0
        }
        for idx in trange(len(city_images), desc='Evaluate_Feature_Distribution '+k, leave=True):
            im_filename = city_images[idx]
            im = np.asarray(Image.open(im_filename))
            # add to sum
            distrib = evaluate_feature_distrib(im)

            for feature_class in total_feature_distrib.keys():
                total_feature_distrib[feature_class] += distrib[feature_ids[feature_class]]

        # calc average
        for feature_class in total_feature_distrib.keys():
            total_feature_distrib[feature_class] /= len(city_images)
        city_distribs[k] = total_feature_distrib

    return city_distribs


def evaluate_feature_distrib_for_fold(gt_datadir="../ANN_Data/de_top14_cropped/gt_data",
                                      prefix_list=['GT_hanno', 'GT_dresd', 'GT_dortm']):

    feature_ids = {
        "unlabelled": 0,
        "building": 1,
        "wood": 2,
        "water": 3,
        "road": 4,
        "residential": 5
    }
    # load images
    datadir = gt_datadir
    masks = []
    old_dir = os.getcwd()
    os.chdir(datadir)
    for mask in glob.glob("*.png"):
        flag = False
        for pre in prefix_list:
            if pre in mask:
                flag = True
                break
        if not flag:
            continue
        mask = os.path.join(datadir, mask)
        masks.append(mask)
    os.chdir(old_dir)

    total_feature_distrib = {
        "wood": 0,
        "building": 0,
        "water": 0,
        "road": 0,
        "unlabelled": 0,
        "residential": 0
    }
    for idx in trange(len(masks), desc='Evaluate_Feature_Distribution_From_GT', leave=True):
        im_filename = masks[idx]
        im = np.asarray(Image.open(im_filename))
        # add to sum
        distrib = evaluate_feature_distrib(im)

        for feature_class in total_feature_distrib.keys():
            total_feature_distrib[feature_class] += distrib[feature_ids[feature_class]]

    # calc average
    for feature_class in total_feature_distrib.keys():
        total_feature_distrib[feature_class] /= len(masks)

    return total_feature_distrib


def evaluate_feature_distrib(image):
    feature_ids = {
        "unlabelled": 0,
        "building": 1,
        "wood": 2,
        "water": 3,
        "road": 4,
        "residential": 5
    }

    if len(image.shape) != 2:
        print('Invalid shape ' + str(image.shape))
    unique, counts = np.unique(image, return_counts=True)
    total_points = image.shape[0] * image.shape[1]
    counts = counts / total_points
    distrib = dict(zip(unique, counts))

    # add not existing classes
    for label_class in feature_ids.keys():
        if feature_ids[label_class] not in distrib.keys():
            distrib[feature_ids[label_class]] = 0

    return distrib


def create_fold_list(datadir, prefix_list, to_folder='.', ds_name='osm'):
    """


    """

    # Load dataset file names
    old_dir = os.getcwd()
    os.chdir(datadir)
    file_names = []
    for file in glob.glob("*.png"):
        file_names.append(file)
    os.chdir(old_dir)

    np.random.shuffle(file_names)

    file_names_filtered = []


    for i in trange(len(file_names)):
        file = file_names[i]
        for pre in prefix_list:
            if file.startswith(pre):
                file_names_filtered.append(file)
                break

    train_val_data = file_names_filtered
    train_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in train_val_data]

    print(len(file_names_filtered))
    # write train file
    if to_folder is not None and os.path.isdir(to_folder):
        with open(to_folder + '/' + ds_name + '_list.txt', 'wt') as f:
            for line in train_data_files:
                x, y = line
                f.write(x + ' ' + y + '\n')
            f.flush()

    return to_folder

def delete_gt_without_pred(datadir, remove=False):
    """
    Removes gt images with missing satellite correspond
    :return:
    """
    gt_dir = datadir+'/gt_data'
    os.chdir(gt_dir)
    del_count = 0
    for gt in glob.glob('*.png'):
        sat_file = datadir+"/"+gt[3:]
        if not os.path.isfile(sat_file):
            print("To Delete "+sat_file)
            del_count = del_count + 1
            if remove:
                os.remove(gt)

    print("%d imaged will be deleted" % del_count)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Missing positionial args: datadir, num_folds")
    folds = int(sys.argv[2])
    datadir = sys.argv[1]

    # prefix_1 = ['stutt', 'duess', 'dresd']
    # prefix_2 = ['dortm', 'frank', 'hambu']
    # prefix_3 = ['leipz', 'nuern', 'berli']
    # prefix_4 = ['munic', 'duisb', 'breme']
    # prefix_5 = ['koeln', 'hanno', 'essen']

    prefix_1 = ['koel', 'dres', 'esse']
    prefix_2 = ['duis', 'stut', 'hann']
    prefix_3 = ['leip', 'muni', 'dort']
    prefix_4 = ['brem', 'berl', 'fran']
    prefix_5 = ['hamb', 'nuer', 'dues']

    eu_prefix_1 = ['berl', 'lond', 'mail', 'buka', 'kiew']
    eu_prefix_2 = ['mosk', 'sama', 'rom_', 'sofi', 'kasa']
    eu_prefix_3 = ['prag', 'wien', 'madr', 'muni', 'char']
    eu_prefix_4 = ['pari', 'belg', 'barc', 'hamb', 'ista']
    eu_prefix_5 = ['buda', 'nisc', 'wars', 'mins', 'st_p']

    # Create fold list with given list of city prefixes for the fold
    # create_fold_list(datadir="G:\Datasets\FINAL_DATASETS\de_top15_cropped",
    #                   prefix_list=prefix_5, ds_name="de_top15_e5")

    # createFoldsByCities(datadir, folds)
    # res = evaluate_feature_distrib_for_fold(gt_datadir=datadir+"/gt_data")

    # Remove unused gt data
    # delete_gt_without_pred(datadir, True)

    # Eval feature distribution for folds
    # distrib = evaluate_feature_distrib_from_GT(datadir="G:\Datasets\Final_Datasets\eu_top25_cropped",
    #                                            img_list="./eu_top25_e1_list.txt")
    # print(distrib)
    # distrib = evaluate_feature_distrib_from_GT(datadir="G:\Datasets\Final_Datasets\eu_top25_cropped",
    #                                            img_list="./eu_top25_e2_list.txt")
    # print(distrib)
    # distrib = evaluate_feature_distrib_from_GT(datadir="G:\Datasets\Final_Datasets\eu_top25_cropped",
    #                                            img_list="./eu_top25_e3_list.txt")
    # print(distrib)
    # distrib = evaluate_feature_distrib_from_GT(datadir="G:\Datasets\Final_Datasets\eu_top25_cropped",
    #                                            img_list="./eu_top25_e4_list.txt")
    # print(distrib)
    # distrib = evaluate_feature_distrib_from_GT(datadir="G:\Datasets\Final_Datasets\eu_top25_cropped",
    #                                            img_list="./eu_top25_e5_list.txt")
    # print(distrib)

    #city_distribs = evaluate_feature_distrib_for_city(gt_datadir="../ANN_Data/de_top15_cropped/gt_data")
    #print(city_distribs)
    # {'leipzig': {'water': 0.05705580605655757, 'unlabelled': 0.16406976731240266, 'road': 0.11598346136717878, 'building': 0.1258078100645736, 'residential': 0.34150532175462023, 'wood': 0.19557783344466725}, 'stuttgart': {'water': 0.0014161211311511908, 'unlabelled': 0.14440433088399013, 'road': 0.1222865954130482, 'building': 0.15027680360721443, 'residential': 0.2742244767312404, 'wood': 0.3073916722333556}, 'munic': {'water': 0.00030089067022934755, 'unlabelled': 0.10966011467379203, 'road': 0.10973195279447787, 'building': 0.1145949621465152, 'residential': 0.25208060008906713, 'wood': 0.41363147962591856}, 'dresden': {'water': 0.002634168336673347, 'unlabelled': 0.13842447116455123, 'road': 0.09954177800044547, 'building': 0.08733408483633932, 'residential': 0.28787275662435985, 'wood': 0.3841927410376308}, 'hannover': {'water': 0.010186767980405255, 'unlabelled': 0.14664124916499663, 'road': 0.10815534958806508, 'building': 0.14965002226675572, 'residential': 0.33334378757515, 'wood': 0.252022823424627}, 'berlin': {'water': 0.022438604987753286, 'unlabelled': 0.13154974949899795, 'road': 0.10533592184368727, 'building': 0.15895236584279673, 'residential': 0.393404664885326, 'wood': 0.18831869294143855}, 'frankfurt': {'water': 0.013453033845468712, 'unlabelled': 0.12232456023157418, 'road': 0.11413675684702743, 'building': 0.1543818136272545, 'residential': 0.2853934591405029, 'wood': 0.31031037630817165}, 'dortmund': {'water': 0.0065052827877978186, 'unlabelled': 0.1737399465597863, 'road': 0.10019461701180142, 'building': 0.12257145958583832, 'residential': 0.3247836283678466, 'wood': 0.2722050656869295}, 'duesseldorf': {'water': 0.04999829659318637, 'unlabelled': 0.17276359385437537, 'road': 0.11165811623246501, 'building': 0.16822607437096418, 'residential': 0.31709000222667577, 'wood': 0.1802639167223335}, 'hamburg': {'water': 0.050306418392340256, 'unlabelled': 0.17645556112224442, 'road': 0.10174067022934749, 'building': 0.1509726564239589, 'residential': 0.382320418615008, 'wood': 0.13820427521710094}, 'duisburg': {'water': 0.047585986733001656, 'unlabelled': 0.18495077588249229, 'road': 0.11631329661217711, 'building': 0.16808374200426435, 'residential': 0.3439588959962094, 'wood': 0.1391073027718551}, 'essen': {'water': 0.02824707192162102, 'unlabelled': 0.14968667891338233, 'road': 0.12030683032732135, 'building': 0.14212011801380542, 'residential': 0.3731896849254066, 'wood': 0.18644961589846368}, 'bremen': {'water': 0.02225327321309286, 'unlabelled': 0.1848189156089956, 'road': 0.12843124025829436, 'building': 0.18166749053662887, 'residential': 0.3877347417056337, 'wood': 0.09509433867735474}, 'nuernberg': {'water': 0.004691739033622801, 'unlabelled': 0.15615587842351372, 'road': 0.126414885326208, 'building': 0.18438174682698738, 'residential': 0.32738044978846575, 'wood': 0.20097530060120244}, 'koeln': {'water': 0.04324748385660209, 'unlabelled': 0.15040459251837013, 'road': 0.09438683478067242, 'building': 0.12818426297038524, 'residential': 0.27083736918281004, 'wood': 0.3129394566911603}}
    city_distribs = evaluate_feature_distrib_for_city(gt_datadir="G:/Datasets/FINAL_DATASETS/eu_top25_cropped/gt_data")
    print(city_distribs)
    # {'kiew': {'wood': 0.4161556389327714, 'road': 0.10056768913463225, 'unlabelled': 0.17264828857293302, 'water': 0.07389142092329297, 'building': 0.09027585571353339, 'residential': 0.14646110672283663}, 'mailand': {'wood': 0.12936707752768842, 'road': 0.13332438370846725, 'unlabelled': 0.22411599231868531, 'water': 0.0036888531618435127, 'building': 0.23371682743837058, 'residential': 0.2757868658449448}, 'budapest': {'wood': 0.28093019296951816, 'road': 0.11436933997050151, 'unlabelled': 0.14424401425762043, 'water': 0.0029628441494591934, 'building': 0.17746379056047204, 'residential': 0.28002981809242855}, 'minsk': {'wood': 0.2557410818713449, 'road': 0.13584185903354884, 'unlabelled': 0.2145803939673745, 'water': 0.048429316712834715, 'building': 0.11708598030163127, 'residential': 0.22832136811326567}, 'kasan': {'wood': 0.06422631637661758, 'road': 0.12659712182061578, 'unlabelled': 0.2159189089692103, 'water': 0.12553255243195002, 'building': 0.1531366912092816, 'residential': 0.314588409192325}, 'charkiw': {'wood': 0.4114804545454542, 'road': 0.08973225589225585, 'unlabelled': 0.20285044893378226, 'water': 0.02367941638608305, 'building': 0.11023688552188564, 'residential': 0.16202053872053876}, 'prag': {'wood': 0.5202001262626266, 'road': 0.092898970959596, 'unlabelled': 0.22791936868686885, 'water': 0.016695650252525254, 'building': 0.08149719696969697, 'residential': 0.060788686868686846}, 'london': {'wood': 0.12402612286890063, 'road': 0.13444233392122287, 'unlabelled': 0.19045192239858932, 'water': 0.04674236037624927, 'building': 0.2072945208700764, 'residential': 0.297042739564962}, 'barcelona': {'wood': 0.42419893939393943, 'road': 0.11519689393939392, 'unlabelled': 0.21064454545454545, 'water': 0.0015134722222222223, 'building': 0.21120968434343448, 'residential': 0.03723646464646464}, 'hamburg': {'wood': 0.13746782259966983, 'road': 0.10125533734371304, 'unlabelled': 0.17755550247699922, 'water': 0.05202346661948575, 'building': 0.15073701344656754, 'residential': 0.3809608575135647}, 'paris': {'wood': 0.1257192673992673, 'road': 0.13372442307692323, 'unlabelled': 0.06965909035409036, 'water': 8.654456654456655e-05, 'building': 0.2880231043956046, 'residential': 0.38278757020757065}, 'bukarest': {'wood': 0.06173823345817728, 'road': 0.14394609862671653, 'unlabelled': 0.03907812109862671, 'water': 0.0, 'building': 0.23390874687890142, 'residential': 0.5213287999375783}, 'nischni_nowgorod': {'wood': 0.07208014048531292, 'road': 0.14195257822477647, 'unlabelled': 0.24014583333333356, 'water': 0.04737652458492976, 'building': 0.17918338122605365, 'residential': 0.3192615421455937}, 'madrid': {'wood': 0.23048596096096116, 'road': 0.17967824074074037, 'unlabelled': 0.16639200450450445, 'water': 0.0024961586586586576, 'building': 0.21029095970970957, 'residential': 0.2106566754254254}, 'st_petersburg': {'wood': 0.1657028841245534, 'road': 0.16108467542964075, 'unlabelled': 0.2107680640632978, 'water': 0.0814950740173559, 'building': 0.1695483239748169, 'residential': 0.21140097839033514}, 'wien': {'wood': 0.17598641287527084, 'road': 0.12793948467966576, 'unlabelled': 0.17942674868461775, 'water': 0.04145593082636953, 'building': 0.21573835886722384, 'residential': 0.259453064066852}, 'moskau': {'wood': 0.17619929039659143, 'road': 0.14859764503441475, 'unlabelled': 0.20719112913798782, 'water': 0.020240221238938053, 'building': 0.16513091281547015, 'residential': 0.2826408013765979}, 'istanbul': {'wood': 0.3773206505071948, 'road': 0.11234363941967462, 'unlabelled': 0.27707076256192514, 'water': 0.04237619426751592, 'building': 0.16275271880160422, 'residential': 0.0281360344420854}, 'belgrad': {'wood': 0.33256997892201134, 'road': 0.08512808641975309, 'unlabelled': 0.2275244354110206, 'water': 0.10735883769948812, 'building': 0.10460812255344777, 'residential': 0.14281053899427887}, 'warschau': {'wood': 0.24659035181236677, 'road': 0.14126459369817582, 'unlabelled': 0.1777814321250889, 'water': 0.004905751006870409, 'building': 0.14027014925373135, 'residential': 0.28918772210376686}, 'munic': {'wood': 0.41681502267573706, 'road': 0.10923769274376419, 'unlabelled': 0.10840848072562358, 'water': 0.0003064172335600907, 'building': 0.1134388151927437, 'residential': 0.25179357142857156}, 'samara': {'wood': 0.1369516987179487, 'road': 0.09573124999999993, 'unlabelled': 0.2095779487179488, 'water': 0.11764892094017095, 'building': 0.15502631410256407, 'residential': 0.28506386752136775}, 'rom': {'wood': 0.2461802906818799, 'road': 0.12352319342569969, 'unlabelled': 0.18596800461831034, 'water': 0.0017813977180114093, 'building': 0.2031498234175498, 'residential': 0.2393972901385492}, 'berlin': {'wood': 0.18662627032520332, 'road': 0.10508517389340549, 'unlabelled': 0.13250116869918696, 'water': 0.022604025519421866, 'building': 0.1587027100271003, 'residential': 0.3944806515356818}, 'sofia': {'wood': 0.1615091104497354, 'road': 0.1412565228174603, 'unlabelled': 0.16760161210317456, 'water': 0.0007438244047619047, 'building': 0.18489391534391542, 'residential': 0.3439950148809523}}

    # todo plot!
