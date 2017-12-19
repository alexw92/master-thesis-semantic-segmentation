# source: https://gist.github.com/robintw/9366322#file-osm_to_shapefile-py

import xml.etree.cElementTree as ET
from shapely.geometry import mapping, Polygon
import fiona		# install fiona https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona requires gdal!!
import os
import matplotlib.pyplot as plt
import shapefile    # pip install pyshp
import cv2
import numpy as np
import math
from Polygon_Calc import color_polygon


# left, bottom, right, top
# bounds = b.attrib['minlat'], b.attrib['minlon'], b.attrib['maxlat'], b.attrib['maxlon']
def __convert_longLat_to_pixel(minX, minY, maxX, maxY, x, y, newXMax=300, newYMax=300):
    """
    :param minX: minLat
    :param minY: minLong
    :param maxX: maxLat
    :param maxY: maxLong
    :param x:    x (Lat) to be transformed
    :param y:    y (Long) to be transformed
    :param newXMax: new XMax value (min is always 0)
    :param newYMax: new YMax value (min is always 0)
    :return:
    """
    xdiff = maxX - minX
    ydiff = maxY - minY
    xscale = newXMax/xdiff
    yscale = newYMax/ydiff
    resx = xscale * (x-minX)
    resy = yscale * (y-minY)
    return resx, resy


# https://forum.openstreetmap.org/viewtopic.php?id=4353
# sollte eig klappen.... aber iwie manchmal negative werte
def __convert_longLat_to_pixel2(latmin, lonmin, latmax, lonmax, lon, lat, maxpixel):
    """
    :param newXMax: new XMax value (min is always 0)
    :param newYMax: new YMax value (min is always 0)
    :return:
    """
    if lon < lonmin or lon > lonmax:
        # print('lon out of bounds')
     pass
    if lat < latmin or lat > latmax:
        # print('lat out of bounds')
        pass
    resx = int((lon - lonmin)/(lonmax-lonmin)*maxpixel)
    resy = int((lat-latmin)/(latmax-latmin)*maxpixel)
    return resx, resy


# Equal to the kaggle color poly function
def colorPolys(polygons, im_size, poly_type, img_mask=None):
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
    cv2.fillPoly(img_mask, exteriors, poly_type)    # color polygons white
    cv2.fillPoly(img_mask, interiors, 0)            # leave holes black lets hope there are no holes
    return img_mask


# download maps from: https://www.openstreetmap.org/export#map=16/49.7513/9.9609
input_filename = "../ANN_DATA/wuerzburg_klein.osm"
output_filename = "../ANN_DATA/wuerzburg_klein.shp"

# Parse the XML from the OSM file
tree = ET.ElementTree(file=input_filename)

# Get the root of the XML (the <osm> node)
r = tree.getroot()


# Get all of the individual points from the file into a dictionary,
# with the keys being the ids of the nodes
nodes = {}

# get bounds
b = r.findall('bounds')[0]

# left, bottom, right, top
bounds = float(b.attrib['minlat']), float(b.attrib['minlon']), float(b.attrib['maxlat']), float(b.attrib['maxlon'])
latmin, lonmin, latmax, lonmax = bounds
print('bounds:' + str(bounds))
  # lat="49.7551035" lon="9.9579631"

# Put all nodes in a dictionary
for n in r.findall('node'):
    nodes[n.attrib['id']] = (float(n.attrib['lon']), float(n.attrib['lat']))


# Create a dictionary to hold the polygons we create
polygons = {}
poly_list_building = []
poly_list_wood = []
# count building for test purposes
n_buildings = 0
pixelwh = 1500

# For each 'way' in the file
for way in r.findall("way"):

    coords = []
    valid = True
    isBuilding = False
    isWood = False
    i = 0
    # Look at all of the children of the <way> node
    for c in way.getchildren():
        if c.tag == "nd":
            # If it's a <nd> tag then it refers to a node, so get the lat-lon of that node
            # using the dictionary we created earlier and append it to the co-ordinate list
            ll = nodes[c.attrib['ref']]  # ll = list of lon,lat
            lon, lat = ll
            if lon<lonmin or lon>lonmax or lat<latmin or lat>latmax:
                # print('coord out of bounds')
                valid = False
            x, y = __convert_longLat_to_pixel2(latmin, lonmin, latmax, lonmax, lon, lat, pixelwh)  # lat="49.7551035" lon="9.9579631"
            ll = x, y
            # print(ll)
            coords.append(ll)
        if c.tag == "tag":
            # If it's a <tag> tag (confusing, huh?) then it'll have some useful information in it for us
            # Get out the information we want as an attribute (in this case village name - which could be
            # in any of a number of fields, so we check)
            village_name = str(i)
            i = i+1
            if c.attrib['k'] == 'building':
                isBuilding = True
            elif c.attrib['k'] == 'natural' and c.attrib['v'] == 'wood':
                isWood = True
    # Take the list of co-ordinates and convert to a Shapely polygon
    if len(coords) > 2 and isBuilding:
        n_buildings = n_buildings + 1
        poly_list_building.append(Polygon(coords))
    if len(coords) > 2 and isWood:
        poly_list_wood.append(Polygon(coords))

print('buildings: '+str(n_buildings))

# color polys, polytype = color
# label buildings
img_mask = colorPolys(polygons=poly_list_building, im_size=(pixelwh, pixelwh), poly_type=1)
# label woods
img_mask = colorPolys(polygons=poly_list_wood, im_size=(pixelwh, pixelwh), poly_type=2, img_mask=img_mask)
train_mask = img_mask
plt.figure()
# plt.axis('equal')
print(train_mask.shape)
# plt.title('image: \''+sample_imgid+'\''+' Classes = '+str(classlist))
# plt.xlabel(str(xstart)+' - '+str(xend))
# plt.ylabel(str(ystart)+' - '+str(yend))
plt.imshow(train_mask,vmin=0, vmax=10, cmap='tab10', origin='lower')
plt.show()
