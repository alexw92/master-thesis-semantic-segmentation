
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import gdal                     # link to gdal whl in ../mixed
import csv
from shapely.geometry import MultiPolygon, Polygon  # link to gdal whl in ../mixed
import shapely.wkt # shapely tut https://toblerity.org/shapely/manual.html#polygons
import shapely.affinity
import cv2
import numpy as np

# fixed error _csv.Error: field larger than field limit (131072) in csv.reader(open('...'))
csv.field_size_limit(2147483647)  # sys.maxsize seems to be to big


# get env
prefix = 'G:/'
with open('../ANN_DATA/ENV') as env_file:
    env = env_file.readline()
if env == 'PC2':
    prefix = '../ANN_DATA/'
elif env == 'Main':
    prefix = 'G:/'

image_path = prefix+'ThreeBand_Images'  # 6010_0_0.tif
grid_sizes = '../ANN_DATA/DSTL SIFD/grid_sizes.csv'
train_polys = '../ANN_DATA/DSTL SIFD/train_wkt_v4.csv'
sample_imgid = '6120_2_2'

# kaggle data attributes
# getting number of tif-images = 450, windows batch:
# set i=0
# for %%a in (*.tif) do set /a i+=1
# better: dir *.tif   (https://superuser.com/a/345620)

PolygonType = {'BUILDING': '1', 'MIS_STRUCTURES': '2', 'ROAD': '3', 'TRACK': '4', 'TREES': '5', 'CROPS': '6',
               'WATERWAY': '7', 'STANDING_WATER': '8', 'VEHICLE_LARGE': '9', 'VEHICLE_SMALL': '10'}


def __load_gridsize(img_id):
    # Load grid size

    _x_max = _y_min = None
    for _im_id, _x, _y in csv.reader(open(grid_sizes)):
        if _im_id == img_id:
            _x_max, _y_min = float(_x), float(_y)
            break
    return _x_max, _y_min


def __load_polygons(img_id, poly_type):
    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open('../ANN_DATA/DSTL SIFD/train_wkt_v4.csv')):
        if _im_id == img_id and _poly_type == poly_type:
            train_polygons = shapely.wkt.loads(_poly)
            break
    return train_polygons


def __get_scalers(im_size):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


#  https://toblerity.org/shapely/manual.html#polygons
def __mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)    # color polygons white
    cv2.fillPoly(img_mask, interiors, 0)    # leave holes black
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


#  data processing tutorial from kaggle
#  https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#data-processing-tutorial
class_name = 'TRACK'
xstart, xend = 2900, 3200
ystart, yend = 2000, 2300

x_max, y_min = __load_gridsize(sample_imgid)
train_polys = __load_polygons(sample_imgid, PolygonType[class_name])


im_rgb = gdal.Open(image_path+"/"+sample_imgid+".tif").ReadAsArray().transpose([1, 2, 0])

imsize = im_rgb.shape[:2]  # first two elements width and height
x_scaler, y_scaler = __get_scalers(imsize)

train_polygons_scaled = shapely.affinity.scale(
    train_polys, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


train_mask = __mask_for_polygons(train_polygons_scaled, imsize)

img = (__scale_percentile(im_rgb[xstart:xend, ystart:yend]))
plt.figure()  # https://stackoverflow.com/a/41210974/8862202,  to show multiple figures
plt.title('image: \''+sample_imgid+'\'')
plt.xlabel(str(xstart)+' - '+str(xend))
plt.ylabel(str(ystart)+' - '+str(yend))
plt.imshow(img)
train_mask = train_mask[xstart:xend, ystart:yend]
plt.figure()
plt.title('image: \''+sample_imgid+'\''+' Class = '+class_name)
plt.xlabel(str(xstart)+' - '+str(xend))
plt.ylabel(str(ystart)+' - '+str(yend))
plt.imshow(train_mask, cmap='Greys')
plt.show()
