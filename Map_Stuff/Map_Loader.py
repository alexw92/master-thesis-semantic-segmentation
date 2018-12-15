# python example of dl                 -> https://stackoverflow.com/a/39574473/8862202
# visibility of labels off  googleapi  -> 'style=feature:all|element:labels|visibility:off'

# Google Maps Params & docs -> https://developers.google.com/maps/documentation/static-maps/intro
from io import BytesIO
from PIL import Image  # pip install pillow (for python3)
from urllib import request
import matplotlib as mpl
import matplotlib.pyplot as plt  # this is if you want to plot the map using pyplot pip install matplotlib
import numpy as np
from scipy.ndimage import zoom, imread
import Approach2 as helper
import xml.etree.cElementTree as ET
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
import shapely
import cv2
import random
import json
import time
import os
from pprint import pprint

# google api key
API_KEY = 'AIzaSyDGB5AkLLgzXB6_rUwVjWIUqQ1FT3sXzO0'
file_list = []
initialized = False
current_index = 0
pixel_offset_google = 20  # the number of pixel that will be added additionally and then cut


def initialize(im_width=224, im_height=224):
    f_list = os.listdir('..\ANN_DATA\Google_Osm\wuerzburg_'+str(im_width)+'_'+str(im_height))
    osm_list = os.listdir('..\ANN_DATA\Google_Osm\wuerzburg_'+str(im_width)+'_'+str(im_height)+'\osm')
    global initialized
    initialized = True
    for i in range(0, len(f_list)):
        file_name = f_list[i]
        if not os.path.isdir('../ANN_DATA/Google_Osm/wuerzburg_'+str(im_width)+'_'+str(im_height) + '/' + file_name):
            coords = file_name.split('.png')[0].split('_')
            osm_file = file_name.replace('.png', '.osm')
            file_osmfile_lat_long = file_name, osm_file, float(coords[1]), float(coords[2])
            file_list.append(file_osmfile_lat_long)


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
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]
    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat, maxwidth, maxheight):
    """
    :param newXMax: new XMax value (min is always 0)
    :param newYMax: new YMax value (min is always 0)
    :return:
    """
    if lon < lonmin or lon > lonmax:
        print('lon out of bounds')
    if lat < latmin or lat > latmax:
        print('lat out of bounds')
    resx = int(((lon - lonmin) / (lonmax - lonmin)) * maxwidth)
    resy = int(((lat - latmin) / (latmax - latmin)) * maxheight)
    return resx, resy


def readPolygons(xmlroot, bbox, pxwidth, pxheight):
    """

    :param xmlroot: root object of the xml structure containing osm data
    :param bbox: the bounding box (lonmin, latmin, lonmax, latmax)
    :param pxwidth: current version requires pxwidth = pxheight
    :param pxheight: current version requires pxwidth = pxheight
    :return:
    """
    # Polygon WS-ES-EN-WN-WS
    bbox_w, bbox_s, bbox_e, bbox_n = bbox
    ext =[(bbox_w, bbox_s),
          (bbox_w, bbox_n),
          (bbox_e, bbox_n),
          (bbox_e, bbox_s),
          (bbox_w, bbox_s)]
        # [(bbox_w, bbox_s),
        #    (bbox_e, bbox_s),
        #    (bbox_e, bbox_n),
        #    (bbox_w, bbox_n),
        #    (bbox_w, bbox_s)]

    poly_bbox = Polygon(ext)
    lonmin, latmin, lonmax, latmax = bbox
    poly_list_buildingll = []
    poly_list_wood = []
    poly_list_water = []
    poly_list_waterll = []
    poly_list_road_residential = []
    poly_list_road_footway = []
    poly_list_road_secondary = []
    poly_list_road_tertiary = []
    poly_list_road_service = []
    poly_list_road_path = []
    poly_list_road_track = []
    poly_list_road_unclass = []
    # count building for test purposes
    nodes = {}
    ways = {}
    # Put all nodes in a dictionary
    for n in xmlroot.findall('node'):
        nodes[n.attrib['id']] = (float(n.attrib['lon']), float(n.attrib['lat']))

    # For each 'way' in the file
    for way in xmlroot.findall("way"):
        ways[way.attrib['id']] = []
        coordsll = []
        valid = True
        isBuilding = False
        isWood = False
        isRoadResidential = False
        isRoadFootway = False
        isRoadSecondary = False
        isRoadTertiary = False
        isRoadService = False
        isTrackService = False
        isPathService = False
        isWater = False
        isUnclass = False
        width = -1
        for c in way.getchildren():
            if c.tag == "nd":
                # If it's a <nd> tag then it refers to a node, so get the lat-lon of that node
                # using the dictionary we created earlier and append it to the co-ordinate list
                ways[way.attrib['id']].append(c.attrib['ref'])  # add reference to node to way

                ll = nodes[c.attrib['ref']]  # ll = list of lon,lat
                lon, lat = ll
                coordsll.append((lon, lat))
            if c.tag == "tag":
                # If it's a <tag> tag (confusing, huh?) then it'll have some useful information in it for us
                # Get out the information we want as an attribute (in this case village name - which could be
                # in any of a number of fields, so we check)
                if c.attrib['k'] == 'building':
                    isBuilding = True
                elif c.attrib['k'] == 'natural' and c.attrib['v'] == 'wood' or \
                        c.attrib['k'] == 'landuse' and c.attrib['v'] == 'forest' or \
                        c.attrib['k'] == 'landcover' and c.attrib['v'] == 'trees':
                    isWood = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'residential':
                    isRoadResidential = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'footway':
                    isRoadFootway = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'secondary':
                    isRoadSecondary = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'tertiary':
                    isRoadTertiary = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'service':
                    isRoadService = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'track':
                    isTrackService = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'path':
                    isPathService = True
                elif c.attrib['k'] == 'highway' and c.attrib['v'] == 'unclassified':
                    isUnclass = True
                elif c.attrib['k'] == 'waterway' and c.attrib['v'] == 'river':
                    isWater = True
                elif c.attrib['k'] == 'width':
                    width = c.attrib['v']
                    # Take the list of co-ordinates and convert to a Shapely polygon
        if len(coordsll) > 2 and isBuilding:
            poly_list_buildingll.append(Polygon(coordsll))
        if len(coordsll) > 2 and isWood:
            poly_list_wood.append(Polygon(coordsll))
        if len(coordsll) > 2 and isRoadResidential:
            poly_list_road_residential.append(LineString(coordsll))
        if len(coordsll) > 2 and isRoadFootway:
            poly_list_road_footway.append(LineString(coordsll))
        if len(coordsll) > 2 and isRoadSecondary:
            poly_list_road_secondary.append(LineString(coordsll))
        if len(coordsll) > 2 and isRoadTertiary:
            poly_list_road_tertiary.append(LineString(coordsll))
        if len(coordsll) > 2 and isRoadService:
            poly_list_road_service.append(LineString(coordsll))
        if len(coordsll) > 2 and isPathService:
            poly_list_road_path.append(LineString(coordsll))
        if len(coordsll) > 2 and isTrackService:
            poly_list_road_track.append(LineString(coordsll))
        if len(coordsll) > 2 and isUnclass:
            poly_list_road_unclass.append(LineString(coordsll))
        if len(coordsll) > 2 and isWater and float(width) > 0:
            poly_list_water.append((LineString(coordsll), float(width)))
    # for each relation in file
    for relation in xmlroot.findall("relation"):
        polylist_relation = []
        isWater = False
        for c in relation.getchildren():
            if c.tag == "member" and c.attrib['type'] == 'way':
                way_ref = c.attrib['ref']
                # add way ref only if it exists
                if way_ref in ways.keys():
                    polylist_relation.append(way_ref)
            if c.tag == "tag":
                if c.attrib['k'] == 'type' and c.attrib['v'] == 'waterway' or \
                        c.attrib['k'] == 'natural' and c.attrib['v'] == 'water' or \
                        c.attrib['k'] == 'water' and c.attrib['v'] == 'lake' or \
                        c.attrib['k'] == 'waterway' and c.attrib['v'] == 'river':
                    isWater = True
        # if water relation
        if len(polylist_relation) > 0 and isWater:
            # todo add polygons to poly_list_water
            for member in polylist_relation:
                way = ways[member]
                coordsll = []
                for c in way:
                    ll = nodes[c]  # ll = list of lon,lat
                    lon, lat = ll
                    coordsll.append((lon, lat))
                # add each member of the relation to the list
                if len(coordsll) > 2:
                    poly_list_waterll.append(Polygon(coordsll))
                print(len(poly_list_waterll))

    # print(len(poly_list_waterll))
    # print(len(poly_list_wood))
    # print(len(poly_list_buildingll))
    # print(len(poly_list_road_residential))
    # !!! Intersect Polygons with bbox to handle with overlapping ones !!!

    # FIX BUILDINGS
    fixed_polys_building = []
    for p in poly_list_buildingll:

        inter = poly_bbox.intersection(p)
        if isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for sub_poly in inter:
                poly = []
                for lon, lat in sub_poly.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                fixed_polys_building.append(Polygon(poly))
        elif isinstance(inter, shapely.geometry.polygon.Polygon):
            poly = []
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            fixed_polys_building.append(Polygon(poly))
        else:
            print('building: ' + str(inter.__class__()))

    # FIX WATERS
    fixed_polys_water = []
    # FIX SELF INTERSECTING POLYGON thx to:  https://stackoverflow.com/a/20873812/8862202
    for line, waterwidth in poly_list_water:
        transformed_width = (
                                        waterwidth / 12) * 0.00005  # unter der annahme das ein secondary highway 12 meter breit ist
        water_polygon = []
        water_polygon.append(line.buffer(transformed_width))
        water_polygon = cascaded_union(water_polygon)
        inter = poly_bbox.intersection(water_polygon.buffer(0))
        if isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                fixed_polys_water.append(Polygon(poly))
        elif (isinstance(inter, shapely.geometry.polygon.Polygon)):  # Polygon
            poly = []
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            fixed_polys_water.append(Polygon(poly))
        else:  # ??
            print('water: ' + str(inter.__class__()))

    # FIX WOODS
    fixed_polys_woods = []
    # FIX SELF INTERSECTING POLYGON thx to:  https://stackoverflow.com/a/20873812/8862202
    for p in poly_list_wood:
        inter = poly_bbox.intersection(p.buffer(0))
        if isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                fixed_polys_woods.append(Polygon(poly))
        elif isinstance(inter, shapely.geometry.polygon.Polygon):  # Polygon
            poly = []
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            fixed_polys_woods.append(Polygon(poly))
        else:  # ??
            print('wood:' + str(inter.__class__()))

    # FIX ROADS Residential (transform lines strings to polygons and transform coords from lon/lat to pixel)
    fixed_polys_roads = []
    road_width_residential = 0.000035
    for line in poly_list_road_residential:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_residential))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Footway (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_footway = 0.000015
    for line in poly_list_road_footway:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_footway))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Secondary (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_secondary = 0.00005
    for line in poly_list_road_secondary:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_secondary))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Tertiary (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_tertiary = 0.000035
    for line in poly_list_road_tertiary:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_tertiary))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Service (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_service = 0.000025
    for line in poly_list_road_service:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_service))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Track (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_track = 0.00001
    for line in poly_list_road_track:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_track))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Path (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_path = 0.00001
    for line in poly_list_road_path:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_path))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    # FIX ROADS Unclassified (transform lines strings to polygons and transform coords from lon/lat to pixel)
    road_width_unclass = 0.00001
    for line in poly_list_road_unclass:
        road_polygon = []
        road_polygon.append(line.buffer(road_width_unclass))
        road_polygon = cascaded_union(road_polygon)
        # check intersection with bbox
        inter = poly_bbox.intersection(road_polygon)
        poly = []
        if isinstance(inter, shapely.geometry.polygon.Polygon):
            for lon, lat in inter.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            roadp = Polygon(poly)
            fixed_polys_roads.append(roadp)
        elif isinstance(inter, shapely.geometry.multipolygon.MultiPolygon):
            for p_int in inter:
                poly = []
                for lon, lat in p_int.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                roadp = Polygon(poly)
                fixed_polys_roads.append(roadp)

    polygon_list = {
        'buildingll': fixed_polys_building,
        'waterll': fixed_polys_water,
        'woodll': fixed_polys_woods,
        'roadll': fixed_polys_roads
    }
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

    w, h = im_size
    im_size = h,w
    if img_mask is None:
        img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    # cv2.fillPoly(img_mask, exteriors, poly_type) Ist wohl ein BUG in opencv: river - siehe bilder, Straßenlöcher!!!
    for ex in exteriors:
        cv2.fillPoly(img_mask, [ex], poly_type)  # color polygons
    cv2.fillPoly(img_mask, interiors, 0)  # leave holes black lets hope there are no holes
    return img_mask


def maybe_download_images(city='wuerzburg', zoom=16, width=224, height=224, datadir='../ANN_DATA/Google_Osm', num_imgs=10):
    dirname = datadir + '/' + city + '_' + str(width)+'_'+str(height)
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname_osm = dirname + '/' + 'osm'
    if not os.path.isdir(dirname_osm):
        os.mkdir(dirname_osm)

        # google load data
        coords = []
        json_data = ''
        with open('city_bounds.json', 'r') as f:
            json_data = json.loads(f.read())
        # pprint(json_data)
        lat_left = json_data[city]['south']
        lat_right = json_data[city]['north']
        lon_left = json_data[city]['west']
        lon_right = json_data[city]['east']


        for i in range(0, num_imgs):
            # thx to  https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range
            map_lat = random.uniform(lat_left, lat_right)
            map_lon = random.uniform(lon_left, lon_right)

            #  print(str(lat_left)+' '+str(lat_right))
            #  print(str(lon_left)+' '+str(lon_right))
            url = 'http://maps.googleapis.com/maps/api/staticmap' + '?' \
                  + 'key=' + API_KEY \
                  + '&center=' + str(map_lat) + ',' \
                  + str(map_lon) + \
                  '&size=' + str(width) + 'x' + str(height) + \
                  '&zoom=' + str(zoom) + \
                  '&sensor=false&maptype=hybrid&style=feature:all|element:labels|visibility:off'
            buffer = BytesIO(request.urlopen(url).read())
            image = Image.open(buffer)
            file_name = city[0:3] + '_' + str(map_lat) + '_' + str(map_lon) + '.png'
            image.save(dirname + '/' + file_name)
            print('Image loaded: ' + str(i + 1))
            # load and save corresponding osm
            # Calc BoundingBox of google image
            centerPoint = helper.G_LatLng(map_lat, map_lon)
            corners = helper.getCorners(centerPoint, zoom, width, height)
            # use bbox to load osm data
            # bbox, order: west, south, east, north (min long, min lat, max long, max lat)
            osm_url = 'http://api.openstreetmap.org/api/0.6/map?bbox=' + str(corners['W']) + ',' + str(corners['S']) + \
                      ',' + str(corners['E']) + ',' + str(corners['N'])
            osm_string = request.urlopen(osm_url).read()
            osm_file_name = city[0:3] + '_' + str(map_lat) + '_' + str(map_lon) + '.osm'
            with open(dirname_osm + '/' + osm_file_name, 'wb') as osmfile:
                osmfile.write(osm_string)


def get_sample(offset=pixel_offset_google, img_width=224, img_height=224):
    if not initialized:
        initialize(im_width=img_width, im_height=img_height)
    # lat = 49.7513
    # lon = 9.9609
    # width = 224
    # height = 224
    # size = '224x224'
    zoomf = 16  # find better solution
    # sensor = 'false'
    # maptype = 'hybrid'
    global current_index
    if current_index == len(file_list) - 1:
        print('epoch end reached')
    next_file, next_osm, next_lat, next_long = file_list[current_index]
    current_index = (current_index + 1) % len(file_list)
    # time_google_start = time.clock()
    # url = 'http://maps.googleapis.com/maps/api/staticmap' + '?' + 'center=' + str(lat) + ',' + str(lon) + \
    #      '&size=' + str(width) + 'x' + str(height) + \
    #      '&zoom=' + str(zoomf) + \
    #      '&sensor=' + sensor + '&maptype=' + maptype + '&style=feature:all|element:labels|visibility:off'

    # buffer = BytesIO(request.urlopen(url).read())
    # time_google = time.clock() - time_google_start
    # print('required time by googlemaps: '+str(time_google))
    image = np.asarray(
        Image.open('../ANN_DATA/Google_Osm/wuerzburg_'+str(img_width)+'_'+str(img_height)+'/' + next_file).convert(
            'RGB'))
    with open('../ANN_DATA/Google_Osm/wuerzburg_'+str(img_width)+'_'+str(img_height)+'/osm/' + next_osm, 'rb') as osm_file:
        osm_string = osm_file.read()
    # Calc BoundingBox
    centerPoint = helper.G_LatLng(next_lat, next_long)
    corners = helper.getCorners(centerPoint, zoomf, img_width, img_height)
    bbox = corners['W'], corners['S'], corners['E'], corners['N']
    # Load osm data using bbox
    # Polygon WS-ES-EN-WN-WS
    # time_osm_start = time.clock()
    # # bbox, order: west, south, east, north (min long, min lat, max long, max lat)
    # osm_url = 'http://api.openstreetmap.org/api/0.6/map?bbox=' + str(corners['W']) + ',' + str(corners['S']) + \
    #           ',' + str(corners['E']) + ',' + str(corners['N'])
    # osm_string = request.urlopen(osm_url).read()
    # time_osm = time.clock() - time_osm_start

    # print('required time by osm '+str(time_osm))
    root = ET.fromstring(osm_string)
    poly_dict = readPolygons(root, bbox, pxwidth=img_width, pxheight=img_height)
    poly_list_building = poly_dict['buildingll']
    poly_list_water = poly_dict['waterll']
    poly_list_wood = poly_dict['woodll']
    poly_list_road = poly_dict['roadll']
    # fill polygons into empty image
    train_mask = colorPolys(polygons=poly_list_building, poly_type=1, im_size=(img_width, img_height))
    train_mask = colorPolys(polygons=poly_list_wood, poly_type=2, im_size=(img_width, img_height), img_mask=train_mask)
    train_mask = colorPolys(polygons=poly_list_water, poly_type=3, im_size=(img_width, img_height), img_mask=train_mask)
    train_mask = colorPolys(polygons=poly_list_road, poly_type=4, im_size=(img_width, img_height), img_mask=train_mask)
    # flip to mask to get same dim as orig image

    # train_mask = np.flip(train_mask, 0)
    # increase quality of plotted satellite, thanks to https://stackoverflow.com/a/46161614/8862202
    # plot satellite image (train) and osm labelled image (label)
    train_mask = np.flip(train_mask, 0)
    # [offline DEBUG]Cut to remove google labels at the bottom

    # image = image[:][:-offset][:]
    # train_mask = train_mask[:][:-offset][:]
#    train_mask = train_mask[:][:-offset][:]

    return image, train_mask


if __name__ == '__main__':
    # width = 224
    # height = 224
    # size = '224x224'
    # zoomf = 16
    # sensor = 'false'
    # maptype = 'hybrid'
    """ 
    # googleapi parameters
    # google static map url: http://maps.googleapis.com/maps/api/staticmap?center=49.7513,9.9609&size=512x512
    # &zoom=17&sensor=false&maptype=hybrid&style=feature:all|element:labels|visibility:off
    # corresponding osm    : https://www.openstreetmap.org/export#map=16/49.7513/9.9609

    #links 16/49.79377/9.89788
    #rechts 16/49.7961/10.0050
    #oben   16/49.80352/9.95350
    #unten  16/49.77555/9.95169
    lat = 49.79377
    lon = 9.89788
    width = 224
    height = 224
    size = '224x224'
    zoomf = 16
    sensor = 'false'
    maptype = 'hybrid'

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
    # print(osm_string)
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
    mpl.rcParams['figure.dpi'] = 150
    # plot satellite image (train) and osm labelled image (label)
    fig, ax = plt.subplots(1, 2)
    plt.title('satellite image')
    ax[0].imshow(image)
    plt.title('osm labelled image')
    ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    # plt.imshow(train_mask, vmin=0, vmax=10, cmap='tab10', origin='lower')
    plt.show()

    # test downloader
    # maybe_download_images(num_imgs=200)
    # img_mask, train_mask = get_sample()
    # test water polygon drawing
    #
    #
    """
    'wue_49.76483814774842_9.955256829012866.png'
    'wue_49.76267927265251_9.96082866534171.png'
    # next_file = 'wue_49.76267927265251_9.96082866534171.png'
    # next_osm = 'wue_49.76267927265251_9.96082866534171.osm'
    # next_lat = 49.76267927265251
    # next_long = 9.96082866534171
    # img_mask = np.asarray(
    #     Image.open('../ANN_DATA/Google_Osm/wuerzburg_224/' + next_file).convert(
    #         'RGB'))
    # with open('../ANN_DATA/Google_Osm/wuerzburg_224/osm/' + next_osm, 'rb') as osm_file:
    #     osm_string = osm_file.read()
    # # Calc BoundingBox
    # centerPoint = helper.G_LatLng(next_lat, next_long)
    # corners = helper.getCorners(centerPoint, zoomf, width, height)
    # bbox = corners['W'], corners['S'], corners['E'], corners['N']
    # Load osm data using bbox

    # time_osm_start = time.clock()
    # # bbox, order: west, south, east, north (min long, min lat, max long, max lat)
    # osm_url = 'http://api.openstreetmap.org/api/0.6/map?bbox=' + str(corners['W']) + ',' + str(corners['S']) + \
    #           ',' + str(corners['E']) + ',' + str(corners['N'])
    # osm_string = request.urlopen(osm_url).read()
    # time_osm = time.clock() - time_osm_start

    # print('required time by osm '+str(time_osm))
    # root = ET.fromstring(osm_string)
    # poly_dict = readPolygons(root, bbox, pxwidth=width, pxheight=height)
    # poly_list_buildingll = poly_dict['buildingll']
    # poly_list_waterll = poly_dict['waterll']
    # poly_list_woodll = poly_dict['woodll']
    # poly_list_roadll = poly_dict['roadll']
    # # check polylist sizes
    # print(str(len(poly_list_buildingll)) + ' buildings found')
    # print(str(len(poly_list_woodll)) + ' wood found')
    # print(str(len(poly_list_waterll)) + ' water found')
    # print(str(len(poly_list_roadll)) + ' road found')

    # in docs einfügen: FIX SELF INTERSECTING POLYGON thx to:  https://stackoverflow.com/a/20873812/8862202
    #  Noch nicht sicher wie gut das klappt...siehe ->
    # in docs einfügen: Probleme bei resultierendem Fluss-Polygon in diesem Bild -> auch mit Gebäuden testen
    # eventuell nur ein osm problem?
    # test ende

    # fill polygons into empty image
    # train_mask = colorPolys(polygons=poly_list_buildingll, poly_type=1, im_size=(width, height))
    # train_mask = colorPolys(polygons=poly_list_woodll, poly_type=2, im_size=(width, height), img_mask=train_mask)
    # train_mask = colorPolys(polygons=poly_list_waterll, poly_type=3, im_size=(width, height), img_mask=train_mask)
    # train_mask = colorPolys(polygons=poly_list_roadll, poly_type=4, im_size=(width, height), img_mask=train_mask)
    # negative
    #  train_mask = colorPolys(polygons=poly_list_water, poly_type=3, im_size=(width, height), img_mask=train_mask)
    # flip to mask to get same dim as orig image
    # train_mask = np.flip(train_mask, 0)
    #
    #
    # end test water polygon drawing
    # mpl.rcParams['figure.dpi'] = 120
    # # plot satellite image (train) and osm labelled image (label)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].set_title('satellite image_wue_test')
    # ax[0].imshow(img_mask)
    # ax[1].set_title('osm labelled image_lol')
    # ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    # # plt.imshow(train_mask, vmin=0, vmax=10, cmap='tab10', origin='lower')
    # plt.show()
    maybe_download_images(num_imgs=10, width=640, height=420)
    for i in range(1, 20):
        #print(img_mask)

        img_mask, train_mask = get_sample(img_width=640, img_height=420,offset=pixel_offset_google)
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('satellite image_wue_test')
        ax[0].imshow(img_mask)
        ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
        ax[1].set_title('osm labelled image ' + str(i))
        plt.show()
    # get wue main city bounds
    # fg = 'http://maps.googleapis.com/maps/api/staticmap?center=49.7873,9.9519&size=512x512&zoom=14&sensor=false&maptype=hybrid&style=feature:all|element:labels|visibility:off'
    # buffer = BytesIO(request.urlopen(fg).read())
    # image = Image.open(buffer)
    # lat = 49.7873
    # lon = 9.9519
    # centerPoint = helper.G_LatLng(lat, lon)
    # corners = helper.getCorners(centerPoint, 14, width, height)
    # bbox = corners['W'], corners['S'], corners['E'], corners['N']
    # print(corners)
