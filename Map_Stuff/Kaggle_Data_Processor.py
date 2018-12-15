
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import gdal                     # link to gdal whl in ../mixed
import csv
from shapely.geometry import MultiPolygon, Polygon  # link to gdal whl in ../mixed
import shapely.wkt # shapely tut https://toblerity.org/shapely/manual.html#polygons
import shapely.affinity
import cv2
import numpy as np
import glob
import os
import glob
import sys
from PIL import Image

# modules for making predictions with sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score
from skimage import io

# fixed error _csv.Error: field larger than field limit (131072) in csv.reader(open('...'))
csv.field_size_limit(2147483647)  # sys.maxsize seems to be to big


# get env
prefix = 'H:/'
with open('../ANN_DATA/ENV') as env_file:
    env = env_file.readline()
if env == 'PC2':
    prefix = '../ANN_DATA/DSTL SIFD/'
elif env == 'Main':
    prefix = 'H:/'

image_path = prefix+'ThreeBand_Images'  # RGB 11bit each channel
image_path = "G:\Vaihingen_Dataset\gt_data_new"
grid_sizes_path= image_path+'/grid_sizes.csv'
train_polys_path = image_path+'/train_wkt_v4.csv'
sample_imgid = '6120_2_2'

all_train_images = ['6110_3_1', '6040_1_0', '6160_2_1', '6150_2_3', '6010_4_4', '6060_2_3', '6090_2_0', '6010_1_2',
                    '6040_1_3', '6010_4_2', '6100_2_2', '6110_4_0', '6110_1_2', '6140_3_1', '6170_0_4', '6120_2_0',
                    '6170_4_1', '6040_2_2', '6120_2_2', '6070_2_3', '6170_2_4', '6100_1_3', '6100_2_3', '6040_4_4',
                    '6140_1_2']

vaihingen_imgs = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area2', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area4',
 'top_mosaic_09cm_area5', 'top_mosaic_09cm_area6','top_mosaic_09cm_area7','top_mosaic_09cm_area8',
 'top_mosaic_09cm_area10','top_mosaic_09cm_area11','top_mosaic_09cm_area12',
 'top_mosaic_09cm_area13',
 'top_mosaic_09cm_area14','top_mosaic_09cm_area15','top_mosaic_09cm_area16','top_mosaic_09cm_area17',
 'top_mosaic_09cm_area20','top_mosaic_09cm_area21',
 'top_mosaic_09cm_area22','top_mosaic_09cm_area23','top_mosaic_09cm_area24',
 'top_mosaic_09cm_area26','top_mosaic_09cm_area27','top_mosaic_09cm_area28','top_mosaic_09cm_area29',
 'top_mosaic_09cm_area30','top_mosaic_09cm_area31','top_mosaic_09cm_area32','top_mosaic_09cm_area33',
 'top_mosaic_09cm_area34','top_mosaic_09cm_area35','top_mosaic_09cm_area37',
 'top_mosaic_09cm_area38']


initialized = False
tif_files = []
image_index = 0
cursor_index = 0
new_image = True
max_cursor_index = 100000
epoch_finished = False
# kaggle data attributes
# getting number of tif-images = 450, windows batch:
# set i=0
# for %%a in (*.tif) do set /a i+=1
# better: dir *.tif   (https://superuser.com/a/345620)

PolygonType = {'BUILDING': '1', 'MIS_STRUCTURES': '2', 'ROAD': '3', 'TRACK': '4', 'TREES': '5', 'CROPS': '6',
               'WATERWAY': '7', 'STANDING_WATER': '8', 'VEHICLE_LARGE': '9', 'VEHICLE_SMALL': '10'}

ClassPriority = ['WATERWAY', 'MIS_STRUCTURES', 'VEHICLE_SMALL',
                 'VEHICLE_LARGE', 'STANDING_WATER', 'CROPS',
                 'TRACK', 'ROAD', 'BUILDING', 'TREES']


def get_sample_vaihingen(img_width=600, img_height=600):
    """
       Returns the next image-ground-truth pair
       When epoch has already been iterated completely, this method returns the
       epoch beginning from the first image again

       See maploader.py for more details
       :return:
       """
    if not initialized:
        initialize()
    # load file
    global new_image
    if new_image:
        new_image = False
        next_image = vaihingen_imgs[image_index] + '.png'
        global im_rgb
        im_rgb = io.imread(next_image)
        print(im_rgb.shape)
        # get satellite img from superfolder
        global sat_img
        sat_img_file = next_image.replace('.png', '.tif')
        sat_img = io.imread("../top/"+sat_img_file)
        global big_img_width, big_img_height
        big_img_width, big_img_height = im_rgb.shape[:2]  # first two elements width and height

    # check cursor_index
    global max_cursor_index
    max_cursor_index = (big_img_height // img_height) * (big_img_width // img_width) - 1
    if cursor_index == max_cursor_index:
        global cursor_index
        cursor_index = 0
        global image_index
        image_index = (image_index + 1) % len(vaihingen_imgs)
        new_image = True
        if image_index == 0:
            print("Epoch finished! Restarting from beginning!")
            global epoch_finished
            epoch_finished = True
    # get cut image bounds
    row_index = cursor_index // (big_img_width // img_width)
    col_index = cursor_index % (big_img_width // img_width)
    row_start = col_index * img_width
    row_end = row_start + img_width
    col_start = row_index * img_height
    col_end = col_start + img_height

    y = im_rgb[col_start:col_end, row_start:row_end]
    x = sat_img[col_start:col_end, row_start:row_end]
    # update cursor index
    global cursor_index
    cursor_index = cursor_index + 1
    # x = x.astype('uint8')
    # y = y.astype('uint8')
    # im_a = Image.fromarray(x)
    # im_b = Image.fromarray(y)
    # im_a.save('./testImage.png')
    # im_b.save('./testGT.png')
    # print('Imgs saved')
    # return image
    return x, y


def get_sample(img_width=600, img_height=600):
    """
    Returns the next image-ground-truth pair
    When epoch has already been iterated completely, this method returns the
    epoch beginning from the first image again

    See maploader.py for more details
    :return:
    """
    if not initialized:
        initialize()
    # load file
    global new_image
    if new_image:
        new_image = False
        next_image = all_train_images[image_index]+'.tif'
        global im_rgb
        im_rgb = gdal.Open(image_path + "/" + next_image).ReadAsArray().transpose([1, 2, 0])
        im_rgb = __scale_percentile(im_rgb)
        global big_img_width, big_img_height
        big_img_width, big_img_height = im_rgb.shape[:2]  # first two elements width and height

        x_max, y_min = __load_gridsize(next_image)
        imsize = big_img_width, big_img_height
        x_scaler, y_scaler = __get_scalers(imsize, x_max, y_min)
        classlist = []
        global train_mask
        train_mask = np.zeros(imsize, np.uint8)

        for classname in ClassPriority:
            print('adding class: ' + classname)
            classlist.append(classname + '(' + PolygonType[classname] + ')')
            train_polys = __load_polygons(next_image, PolygonType[classname])
            train_polygons_scaled = shapely.affinity.scale(
                train_polys, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
            train_mask = __mask_for_polygons(train_polygons_scaled, imsize, PolygonType[classname], train_mask)

    # check cursor_index
    global max_cursor_index
    max_cursor_index = (big_img_height // img_height) * (big_img_width // img_width) - 1
    if cursor_index == max_cursor_index:
        global cursor_index
        cursor_index = 0
        global image_index
        image_index = (image_index + 1) % len(all_train_images)
        new_image = True
        if image_index == 0:
            print("Epoch finished! Restarting from beginning!")
            global epoch_finished
            epoch_finished = True
    # get cut image bounds
    row_index = cursor_index // (big_img_width // img_width)
    col_index = cursor_index % (big_img_width // img_width)
    row_start = col_index * img_width
    row_end = row_start + img_width
    col_start = row_index * img_height
    col_end = col_start + img_height

    x = im_rgb[col_start:col_end, row_start:row_end]
    y = train_mask[col_start:col_end, row_start:row_end]
    # update cursor index
    global cursor_index
    cursor_index = cursor_index + 1
    # x = x.astype('uint8')
    # y = y.astype('uint8')
    # im_a = Image.fromarray(x)
    # im_b = Image.fromarray(y)
    # im_a.save('./testImage.png')
    # im_b.save('./testGT.png')
    # print('Imgs saved')
    # return image
    return x, y


def initialize(img_path=image_path):
    """
    Initializes data, sets up paths, etc.
    See maploader.py for more details

    :param img_path:
    :return:
    """
    os.chdir(img_path)
    for file in glob.glob("*.tif"):
        tif_files.append(file)
    global initialized
    initialized = True


def __load_gridsize(img_id):
    # Load grid size
    if img_id.endswith('.tif'):
        img_id = img_id.replace('.tif', '')
    _x_max = _y_min = None
    for _im_id, _x, _y in csv.reader(open(grid_sizes_path)):
        if _im_id == img_id:
            _x_max, _y_min = float(_x), float(_y)
            break
    return _x_max, _y_min


def __load_polygons(img_id, poly_type):
    if img_id.endswith('.tif'):
        img_id = img_id.replace('.tif', '')
    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open(train_polys_path)):
        if _im_id == img_id and _poly_type == poly_type:
            train_polygons = shapely.wkt.loads(_poly)
            break
    return train_polygons


def __get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


#  https://toblerity.org/shapely/manual.html#polygons
def __mask_for_polygons(polygons, im_size, poly_type, img_mask=None):
    """

    :param polygons: polygons containing the class
    :param im_size: (x, y) size of the image
    :return:
    """
    if img_mask is None:
        img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, int(poly_type))    # color polygons white
    cv2.fillPoly(img_mask, interiors, 0)    # leave holes black lets hope there are no holes
    return img_mask


def __scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def __show_mask(m):
    # hack for nice display
    plt.imshow(255 * np.stack([m, m, m]), cmap='Greys')  # cmap in order to plot black/white


def evaluate_feature_distrib(image, feature_ids):
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


if __name__ == "__main__":
    # initialize(img_path=image_path)
    # print('number of files '+str(len(tif_files)))
    #
    # #  data processing tutorial from kaggle
    # #  https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#data-processing-tutorial
    # class_name = 'BUILDING'
    # xstart, xend = 1000, 1300
    # ystart, yend = 1000, 1300
    #
    # x_max, y_min = __load_gridsize(sample_imgid)
    # train_polys = __load_polygons(sample_imgid, PolygonType[class_name])
    #
    #
    # im_rgb = gdal.Open(image_path+"/"+sample_imgid+".tif").ReadAsArray().transpose([1, 2, 0])
    # imsize = im_rgb.shape[:2]  # first two elements width and height
    # x_scaler, y_scaler = __get_scalers(imsize,x_max, y_min )
    #
    # train_polygons_scaled = shapely.affinity.scale(
    #     train_polys, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    #
    # train_mask = __mask_for_polygons(train_polygons_scaled, imsize, PolygonType[class_name])
    #
    # classlist = []
    # train_mask = np.zeros(imsize, np.uint8)
    #
    # for classname in ClassPriority:
    #     # if classname == 'MIS_STRUCTURES':
    #     #     continue
    #     print('adding class: ' + classname)
    #     classlist.append(classname+'('+PolygonType[classname]+')')
    #     train_polys = __load_polygons(sample_imgid, PolygonType[classname])
    #     train_polygons_scaled = shapely.affinity.scale(
    #         train_polys, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    #     train_mask = __mask_for_polygons(train_polygons_scaled, imsize, PolygonType[classname], train_mask)
    #     # plt.figure()
    #     # plt.imshow(train_mask[xstart:xend, ystart:yend], vmin=0, vmax=10, cmap='ocean')
    # print(classlist)
    # img = (__scale_percentile(im_rgb[xstart:xend, ystart:yend]))

    # 1) CREATE DATASET 600x600
    #
    # dir = 'G:/DSTL_Challenge_Dataset'
    # if len(sys.argv) > 1:
    #     dir = sys.argv[1]+'/DSTL_Challenge_Dataset'
    # if not os.path.exists(dir):
    #     os.mkdir(dir)
    # if not os.path.exists(dir+'/gt_data'):
    #     os.mkdir(dir+'/gt_data')
    # while not epoch_finished:
    #     x, y = get_sample()
    #     x = x * 255
    #     x = x.astype('uint8')
    #     print("index %d of %d, image %d from %d" % (cursor_index, max_cursor_index, image_index, len(all_train_images)))
    #    # fig, ax = plt.subplots(1, 2)
    #     #   plt.figure()  # https://stackoverflow.com/a/41210974/8862202,  to show multiple figures
    #
    #     # save imgs
    #     x_im = Image.fromarray(x)
    #     y_im = Image.fromarray(y)
    #
    #     id = str(image_index)+"_"+str(cursor_index)
    #     x_name = id+'.png'
    #     y_name = 'GT_'+id+'.png'
    #
    #     x_im.save(dir+"/"+x_name)
    #     y_im.save(dir+"/gt_data/"+y_name)

        # ax[0].imshow(x)
        # ax[0].set_title('image')
        # ax[1].imshow(y, vmin=0, vmax=11, cmap='ocean')
        # ax[1].set_title('label')
     #   plt.imshow(y, vmin=0, vmax=11, cmap='ocean')

       # plt.show()
    #1b) CREATE DATASET 600x600 Vaihingen

    # dir = 'G:/Vahingen_Dataset_2'
    # if len(sys.argv) > 1:
    #     dir = sys.argv[1]+'/Vahingen_Dataset_2'
    # if not os.path.exists(dir):
    #     os.mkdir(dir)
    # if not os.path.exists(dir+'/gt_data'):
    #     os.mkdir(dir+'/gt_data')
    # while not epoch_finished:
    #     x, y = get_sample_vaihingen()
    #     #x = x * 255 no need for this dataset at rgbs are already correctly set to max 255
    #     x = x.astype('uint8')
    #     print("index %d of %d, image %d from %d" % (cursor_index, max_cursor_index, image_index, len(vaihingen_imgs)))
    #     fig, ax = plt.subplots(1, 2)
    #     #   plt.figure()  # https://stackoverflow.com/a/41210974/8862202,  to show multiple figures
    #
    #     # save imgs
    #     if x.shape[0] == 600 and x.shape[1] == 600:
    #         x_im = Image.fromarray(x)
    #         y_im = Image.fromarray(y)
    #
    #         id = str(image_index)+"_"+str(cursor_index)
    #         x_name = id+'.png'
    #         y_name = 'GT_'+id+'.png'
    #
    #         x_im.save(dir+"/"+x_name)
    #         y_im.save(dir+"/gt_data/"+y_name)

        # ax[0].imshow(x)
        # ax[0].set_title('image')
        # ax[1].imshow(y, vmin=0, vmax=11, cmap='ocean')
        # ax[1].set_title('label')
        # plt.imshow(y, vmin=0, vmax=11, cmap='ocean')
        #
        # plt.show()

        feature_ids = {'Impervious surfaces': 0, 'Buildings': 1, 'Low_vegetation': 2, 'Tree': 3,
                       'Car': 4, 'Clutter': 5}
        glob_feature_distrib = {x: 0 for x in feature_ids.values()}
        os.chdir('G:\Vahingen_Dataset_2\gt_data')
        counter = 0
        for file in glob.glob("*.png"):
            counter = counter + 1
            im = np.asarray(Image.open(file))
            local_distrib = evaluate_feature_distrib(im, feature_ids=feature_ids)
            # print(glob_feature_distrib)
            # print(local_distrib)
            for x in local_distrib.keys():
                glob_feature_distrib[x] = glob_feature_distrib[x] + local_distrib[x]

        for x in glob_feature_distrib.keys():
            glob_feature_distrib[x] = glob_feature_distrib[x] / counter

        print(glob_feature_distrib)
       #2) Eval feature distrib
    # feature_ids = {'BUILDING': 1, 'MIS_STRUCTURES': 2, 'ROAD': 3, 'TRACK': 4, 'TREES': 5, 'CROPS': 6,
    #                'WATERWAY': 7, 'STANDING_WATER': 8, 'VEHICLE_LARGE': 9, 'VEHICLE_SMALL': 10, 'OTHER':0}
    # glob_feature_distrib = {x:0 for x in feature_ids.values()}
    # os.chdir('G:\Datasets\DSTL_Challenge_Dataset\gt_data')
    # counter = 0
    # for file in glob.glob("*.png"):
    #     counter = counter + 1
    #     im = np.asarray(Image.open(file))
    #     local_distrib = evaluate_feature_distrib(im, feature_ids=feature_ids)
    #     # print(glob_feature_distrib)
    #     # print(local_distrib)
    #     for x in local_distrib.keys():
    #         glob_feature_distrib[x] = glob_feature_distrib[x]+local_distrib[x]
    #
    # for x in glob_feature_distrib.keys():
    #     glob_feature_distrib[x] = glob_feature_distrib[x]/counter
    #
    # print(glob_feature_distrib)


