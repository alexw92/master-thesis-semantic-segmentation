
# python example of dl      -> https://stackoverflow.com/a/39574473/8862202
# visibility of labels off  -> 'style=feature:all|element:labels|visibility:off'

# Google Maps Params & docs -> https://developers.google.com/maps/documentation/static-maps/intro
from io import BytesIO
from PIL import Image   # pip install pillow (for python3)
from urllib import request
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import gdal                     # link to gdal whl in ../mixed
from gdalconst import *
import csv
from shapely.geometry import MultiPolygon, Polygon  # link to gdal whl in ../mixed
import shapely.wkt # shapely tut https://toblerity.org/shapely/manual.html#polygons
import shapely.affinity
import cv2
from enum import Enum
import numpy as np
import sys

# fixed error _csv.Error: field larger than field limit (131072) in csv.reader(open('...'))
csv.field_size_limit(2147483647)

# corresponding osm https://www.openstreetmap.org/export#map=16/49.7513/9.9609
image_path = 'G:/ThreeBand_Images'  # 6010_0_0.tif
sample_tif = '../mixed/sample.tiff'
sample_tif_2 = 'G:/ThreeBand_Images/6010_0_0.tif'
grid_sizes = '../ANN_DATA/DSTL SIFD/grid_sizes.csv'
train_polys = '../ANN_DATA/DSTL SIFD/train_wkt_v4.csv'
sample_imgid = '6120_2_2'

# kaggle data attributes

W = 3349
H = 3391


class PolygonType(Enum):
    BUILDING = '5'


# googleapi parameters
x = '49.7513'
y = '9.9609'
size = '800x800'
zoom = '16'
sensor = 'false'
maptype = 'hybrid'

url = 'http://maps.googleapis.com/maps/api/staticmap'+'?'+'center='+x+','+y+'&size='+size+'&zoom='+zoom +\
      '&sensor=' + sensor + '&maptype=' + maptype + '&style=feature:all|element:labels|visibility:off'


def __convert_to_pixel(x, y, x_max, y_min):
    W_n = W * (W/(W+1))
    x_n = (x/x_max) * W_n

    H_n = H * (H/(H+1))
    y_n = (y/y_min) * H_n

    return x_n, y_n


def __load_gridsize(img_id):
    # Load grid size

    _x_max = _y_min = None
    for _im_id, _x, _y in csv.reader(open(grid_sizes)):
        if _im_id == img_id:
            _x_max, _y_min = float(_x), float(_y)
            break
    return _x_max, _y_min


def __load_polygons(img_id):
    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open('../ANN_DATA/DSTL SIFD/train_wkt_v4.csv')):
        if _im_id == img_id and _poly_type == PolygonType.BUILDING.value:
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


buffer = BytesIO(request.urlopen(url).read())
image = Image.open(buffer)

# Show Using PIL
# image.show()

# Or using pyplot
# plt.imshow(image)
# plt.show()        #  needed in scripts to show the plot

#  data processing tutorial from kaggle
#  https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#data-processing-tutorial

data_sample = gdal.Open(sample_tif, GA_ReadOnly)
boundary = data_sample.RasterXSize, data_sample.RasterYSize, data_sample.RasterCount
tifArr = data_sample.ReadAsArray()

# display sample tif, works in console
f = plt.imread(sample_tif)
# plt.imshow(f)
# plt.show()

#print(__convert_to_pixel(0.000339, -0.004006))

x_max, y_min = __load_gridsize(sample_imgid)
train_polys = __load_polygons(sample_imgid)


im_rgb = gdal.Open(image_path+"/"+sample_imgid+".tif").ReadAsArray().transpose([1, 2, 0])

imsize = im_rgb.shape[:2]
x_scaler, y_scaler = __get_scalers(imsize)

train_polygons_scaled = shapely.affinity.scale(
    train_polys, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


train_mask = __mask_for_polygons(train_polygons_scaled, imsize)

img = (__scale_percentile(im_rgb[2900:3200, 2000:2300]))
plt.figure()  # https://stackoverflow.com/a/41210974/8862202,  to show multiple figures
plt.imshow(img)
train_mask = train_mask[2900:3200, 2000:2300]
plt.figure()
plt.imshow(train_mask, cmap='Greys')
plt.show()
