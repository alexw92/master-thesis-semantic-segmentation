# python example of dl                 -> https://stackoverflow.com/a/39574473/8862202
# visibility of labels off  googleapi  -> 'style=feature:all|element:labels|visibility:off'

# Google Maps Params & docs -> https://developers.google.com/maps/documentation/static-maps/intro
from io import BytesIO
from PIL import Image   # pip install pillow (for python3)
from urllib import request
import gdal
from gdalconst import *
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib

# googleapi parameters
# corresponding osm https://www.openstreetmap.org/export#map=16/49.7513/9.9609
x = '49.7513'
y = '9.9609'
size = '800x800'
zoom = '16'
sensor = 'false'
maptype = 'hybrid'
sample_tif = '../mixed/sample.tiff'

url = 'http://maps.googleapis.com/maps/api/staticmap'+'?'+'center='+x+','+y+'&size='+size+'&zoom='+zoom +\
      '&sensor=' + sensor + '&maptype=' + maptype + '&style=feature:all|element:labels|visibility:off'

buffer = BytesIO(request.urlopen(url).read())
image = Image.open(buffer)

# Show Using PIL
image.show()

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

# print(__convert_to_pixel(0.000339, -0.004006))
