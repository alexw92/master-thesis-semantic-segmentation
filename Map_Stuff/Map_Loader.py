# python example of dl                 -> https://stackoverflow.com/a/39574473/8862202
# visibility of labels off  googleapi  -> 'style=feature:all|element:labels|visibility:off'

# Google Maps Params & docs -> https://developers.google.com/maps/documentation/static-maps/intro
from io import BytesIO
from PIL import Image   # pip install pillow (for python3)
from urllib import request
import gdal
from gdalconst import *
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import numpy as np
from scipy.ndimage import zoom, imread


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]
    print(img.shape[:2])
    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out
# googleapi parameters
# corresponding osm https://www.openstreetmap.org/export#map=16/49.7513/9.9609
x = '49.7513'
y = '9.9609'
size = '800x800'
zoomf = '16'
sensor = 'false'
maptype = 'hybrid'
sample_tif = '../mixed/sample.tiff'

url = 'http://maps.googleapis.com/maps/api/staticmap'+'?'+'center='+x+','+y+'&size='+size+'&zoom='+zoomf +\
      '&sensor=' + sensor + '&maptype=' + maptype + '&style=feature:all|element:labels|visibility:off'

buffer = BytesIO(request.urlopen(url).read())
image = Image.open(buffer)



# Show Using PIL
# image.show()

# Or using pyplot
# plt.imshow(image)
# plt.show()        #  needed in scripts to show the plot

data_sample = gdal.Open(sample_tif, GA_ReadOnly)
boundary = data_sample.RasterXSize, data_sample.RasterYSize, data_sample.RasterCount
tifArr = data_sample.ReadAsArray()
print('Image boundaries: '+str(boundary))
# display sample tif, works in console
f = plt.imread(sample_tif)
# plt.imshow(f)
# plt.show()
img = imread('../ANN_DATA/zoom_sample.png', True)
# print(__convert_to_pixel(0.000339, -0.004006))
zm1 = clipped_zoom(img, 0.5)
zm2 = clipped_zoom(img, 1.5)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img)
ax[1].imshow(zm1)
ax[2].imshow(zm2)
plt.show()