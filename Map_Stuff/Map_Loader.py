# python example of dl                 -> https://stackoverflow.com/a/39574473/8862202
# visibility of labels off  googleapi  -> 'style=feature:all|element:labels|visibility:off'

# Google Maps Params & docs -> https://developers.google.com/maps/documentation/static-maps/intro
from io import BytesIO
from PIL import Image   # pip install pillow (for python3)
from urllib import request
import gdal
from gdalconst import *
import matplotlib as mpl
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import numpy as np
from scipy.ndimage import zoom, imread
import Approach2 as helper
import xml.etree.cElementTree as ET
from shapely.geometry import  Polygon
import cv2



def clipped_zoom(img, zoom_factor, **kwargs):
    """
    Approach one: Apply zoom with the correct zoom_factor on the image to match
    :param img:
    :param zoom_factor:
    :param kwargs:
    :return:
    """

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


def convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat, maxpixel):
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


def readPolygons(xmlroot, bbox, pxwidth, pxheight):
    """

    :param xmlroot: root object of the xml structure containing osm data
    :param bbox: the bounding box (lonmin, latmin, lonmax, latmax)
    :param pxwidth: current version requires pxwidth = pxheight
    :param pxheight: current version requires pxwidth = pxheight
    :return:
    """

    lonmin, latmin, lonmax, latmax = bbox
    polygons = {}
    poly_list_building = []
    poly_list_wood = []
    # count building for test purposes
    n_buildings = 0
    nodes = {}
    # Put all nodes in a dictionary
    for n in xmlroot.findall('node'):
        nodes[n.attrib['id']] = (float(n.attrib['lon']), float(n.attrib['lat']))

    # For each 'way' in the file
    for way in xmlroot.findall("way"):
        coords = []
        valid = True
        isBuilding = False
        isWood = False
        for c in way.getchildren():
            if c.tag == "nd":
                # If it's a <nd> tag then it refers to a node, so get the lat-lon of that node
                # using the dictionary we created earlier and append it to the co-ordinate list
                ll = nodes[c.attrib['ref']]  # ll = list of lon,lat
                lon, lat = ll
                if lon < lonmin or lon > lonmax or lat < latmin or lat > latmax:
                    # print('coord out of bounds')
                    valid = False
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth)  # lat="49.7551035" lon="9.9579631"
                ll = x, y
                # print(ll)
                coords.append(ll)
            if c.tag == "tag":
                # If it's a <tag> tag (confusing, huh?) then it'll have some useful information in it for us
                # Get out the information we want as an attribute (in this case village name - which could be
                # in any of a number of fields, so we check)
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
    polygon_list = {'building': poly_list_building,
                    'wood': poly_list_wood}
    return polygon_list


# Equal to the kaggle color poly function
def colorPolys(polygons, im_size, poly_type, img_mask=None):
    """
    :param polygons: list of  polygons containing the class
    :param im_size: (x, y) size of the image in pixels
    :param poly_type: the type of the polygon, which is the number used to color the polygons
    :param img_mask: (optional) an image mask to be used to color with polygons, will be created if not given
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


if __name__ == '__main__':
    # googleapi parameters
    # google static map url: http://maps.googleapis.com/maps/api/staticmap?center=49.7513,9.9609&size=512x512
    # &zoom=17&sensor=false&maptype=hybrid&style=feature:all|element:labels|visibility:off
    # corresponding osm    : https://www.openstreetmap.org/export#map=16/49.7513/9.9609
    lat = 49.7513
    lon = 9.9609
    width = 512
    height = 512
    size = '512x512'
    zoomf = 17
    sensor = 'false'
    maptype = 'hybrid'
    sample_tif = '../mixed/sample.tiff'

    url = 'http://maps.googleapis.com/maps/api/staticmap'+'?'+'center='+str(lat)+','+str(lon)+\
          '&size='+str(width)+'x'+str(height)+\
          '&zoom='+str(zoomf) +\
          '&sensor=' + sensor + '&maptype=' + maptype + '&style=feature:all|element:labels|visibility:off'

    buffer = BytesIO(request.urlopen(url).read())
    image = Image.open(buffer)

    # Calc BoundingBox
    centerPoint = helper.G_LatLng(lat, lon)
    corners = helper.getCorners(centerPoint, zoomf, width, height)
    bbox = corners['W'], corners['S'], corners['E'], corners['N']
    print(corners)
    # Load osm data using bbox

    # bbox, order: west, south, east, north (min long, min lat, max long, max lat)
    osm_url = 'http://api.openstreetmap.org/api/0.6/map?bbox='+str(corners['W'])+','+str(corners['S'])+\
              ','+str(corners['E'])+','+str(corners['N'])
    osm_string = request.urlopen(osm_url).read()
    print(osm_string)
    root = ET.fromstring(osm_string)
    poly_dict = readPolygons(root, bbox, pxwidth=width, pxheight=height)
    poly_list_building = poly_dict['building']
    poly_list_wood = poly_dict['wood']
    print(str(len(poly_list_building))+' buildings found')
    print(str(len(poly_list_wood))+' wood found')
    # fill polygons into empty image
    train_mask = colorPolys(polygons=poly_list_building, poly_type=1, im_size=(width, height))
    # flip to mask to get same dim as orig image
    train_mask = np.flip(train_mask, 0)
    # increase quality of plotted satellite, thanks to https://stackoverflow.com/a/46161614/8862202
    mpl.rcParams['figure.dpi'] = 250
    # plot satellite image (train) and osm labelled image (label)
    fig, ax = plt.subplots(1, 2)
    plt.title('satellite image')
    ax[0].imshow(image)
    plt.title('osm labelled image')
    ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    # plt.imshow(train_mask, vmin=0, vmax=10, cmap='tab10', origin='lower')
    plt.show()
