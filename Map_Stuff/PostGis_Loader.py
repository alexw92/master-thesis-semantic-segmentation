import psycopg2 as pg  # Für postgresql ansteuerung
from shapely.geometry import Polygon, Point, LineString
import geojson
from shapely.wkt import loads
import shapely
import math
import random
# Using PyCharm (part of the JetBrains suite) you need to define your script directory as Source:
# Right Click > Mark Directory as > Sources Root
import Approach2 as helper
from Map_Loader import colorPolys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image  # pip install pillow (for python3)
from urllib import request
from tqdm import trange
import os
import glob
import re
import json

# Define Connection strings

initialized = False
png_files = []
# plt.rcParams["figure.figsize"] = [20, 12]  # show big images


connect_strings = {
    "main": "dbname=gis user=postgres password=postgres",
    "pc2": "dbname=gis host=192.168.178.29 user=postgres password=postgres",
    "ls6": ""
}

table_names = {

    "malaga": ("POLYGON((-4.524576310773522 36.7584306448112,"
               " -4.524576310773522 36.681650962207854,"
               " -4.2324086471993025 36.681650962207854,"
               " -4.2324086471993025 36.7584306448112,"
               " -4.524576310773522 36.7584306448112))",
               ("malaga_small_planet_osm_line_lines", "malaga_small_planet_osm_polygon_polygons")),
    "berlin": ("POLYGON((13.183786862415506 52.60627116138647,"
               "         13.183786862415506 52.39727845787755, "
               "         13.62735986046238 52.39727845787755, "
               "         13.62735986046238 52.60627116138647, "
               "         13.183786862415506 52.60627116138647))",
               ("berlin_planet_osm_line_lines", "berlin_planet_osm_polygon_polygons")),
    # "unterfranken": ("POLYGON((9.086199142965881 50.07247606571221,"
    #                  "         9.086199142965881 49.66887056979786,"
    #                  "         10.61879191640338 49.66887056979786,"
    #                  "         10.61879191640338 50.07247606571221,"
    #                  "        9.086199142965881 50.07247606571221))",
    #                  ("unterfranken_planet_osm_line_lines", "unterfranken_planet_osm_polygon_polygons")),
    "poznan": ("POLYGON((16.781386363130494 52.54693583746672,"
               "16.781386363130494 52.34017769544545,"
               "17.155608164888314 52.34017769544545,"
               "17.155608164888314 52.54693583746672,"
               "16.781386363130494 52.54693583746672))",
               ("poznan_planet_osm_line_lines", "poznan_planet_osm_polygon_polygons")),
    "lasvegas": ("POLYGON((-114.99880089347494 36.323388547245926,"
                 "-114.99880089347494 35.98686077835177,"
                 "-115.33525719230306 35.98686077835177,"
                 "-115.33525719230306 36.323388547245926,"
                 "-114.99880089347494 36.323388547245926))",
                 ("las_vegas_planet_osm_line_lines", "las_vegas_planet_osm_polygon_polygons")),
    "co_dc": ("POLYGON ((-77.12 38.80000000000001,"
              " -77.12 39,"
              " -76.91 39,"
              " -76.91 38.80000000000001,"
              " -77.12 38.80000000000001))",
              ("city_only_dc_planet_osm_line_lines", "city_only_dc_planet_osm_polygon_polygons")),
    "co_ny": ("POLYGON ((-74.5 40.34,"
              " -74.5 41.09,"
              " -73.51000000000001 41.09,"
              " -73.51000000000001 40.34,"
              " -74.5 40.34))",
              ("city_only_ny_planet_osm_line_lines", "city_only_ny_planet_osm_polygon_polygons")),
    "bremen": ("POLYGON ((8.67039840014742 53.18108265605292,"
               "8.67039840014742 53.012862546544795,"
               "8.990375206788045 53.012862546544795,"
               "8.990375206788045 53.18108265605292,"
               "8.67039840014742 53.18108265605292))",
               ("bremen_planet_osm_line_lines", "bremen_planet_osm_polygon_polygons")),
    "dortmund": ("POLYGON ((7.296116894797416 51.59913732425761,"
                 " 7.296116894797416 51.398660319864234,"
                 " 7.654545849875543 51.398660319864234,"
                 " 7.654545849875543 51.59913732425761,"
                 " 7.296116894797416 51.59913732425761))",
                 ("dortmund_planet_osm_line_lines", "dortmund_planet_osm_polygon_polygons")),
    "dresden": ("POLYGON ((13.579327991946498 51.156130754769094,"
                " 13.579327991946498 50.961924616105534,"
                " 13.869779041751183 50.961924616105534,"
                " 13.869779041751183 51.156130754769094,"
                " 13.579327991946498 51.156130754769094))",
                ("dresden_planet_osm_line_lines", "dresden_planet_osm_polygon_polygons")),
    "duesseldorf": ("POLYGON ((6.656393809063862 51.300066760034724,"
                    " 6.656393809063862 51.146543742436734,"
                    " 6.901526255352923 51.146543742436734,"
                    " 6.901526255352923 51.300066760034724,"
                    " 6.656393809063862 51.300066760034724))",
                    ("duesseldorf_planet_osm_line_lines", "duesseldorf_planet_osm_polygon_polygons")),
    "duisburg": ("POLYGON((6.604229744278693 51.499256100071676,"
                 "6.604229744278693 51.32452109209132,"
                 "6.983258064591193 51.32452109209132,"
                 "6.983258064591193 51.499256100071676,"
                 "6.604229744278693 51.499256100071676))",
                 ("duisburg_planet_osm_line_lines", "duisburg_planet_osm_polygon_polygons")),
    "essen": ("POLYGON ((6.922142649217021 51.489689468383574,"
              " 6.922142649217021 51.380106897105236,"
              " 7.127449656052957 51.380106897105236,"
              " 7.127449656052957 51.489689468383574,"
              " 6.922142649217021 51.489689468383574))",
              ("essen_planet_osm_line_lines", "essen_planet_osm_polygon_polygons")),
    "frankfurt": ("POLYGON ((8.605955419026351 50.17448260524435,"
                  " 8.605955419026351 50.06794132784324,"
                  " 8.850057897053695 50.06794132784324,"
                  " 8.850057897053695 50.17448260524435,"
                  " 8.605955419026351 50.17448260524435))",
                  ("frankfurt_planet_osm_line_lines", "frankfurt_planet_osm_polygon_polygons")),
    "hamburg": ("POLYGON ((9.871168520200555 53.65196410865872,"
                " 9.871168520200555 53.41239712330349,"
                " 10.247450258481805 53.41239712330349,"
                " 10.247450258481805 53.65196410865872,"
                " 9.871168520200555 53.65196410865872))",
                ("hamburg_planet_osm_line_lines", "hamburg_planet_osm_polygon_polygons")),
    "hannover": ("POLYGON ((9.644931705677807 52.47359448216025,"
                 " 9.644931705677807 52.26649149133107,"
                 " 9.93400946446687 52.26649149133107,"
                 " 9.93400946446687 52.47359448216025,"
                 " 9.644931705677807 52.47359448216025))",
                 ("hannover_planet_osm_line_lines", "hannover_planet_osm_polygon_polygons")),
    "koeln": ("POLYGON ((6.823942854006496 51.03372575881434,"
              " 6.823942854006496 50.822525169082155,"
              " 7.196791364748683 50.822525169082155,"
              " 7.196791364748683 51.03372575881434,"
              " 6.823942854006496 51.03372575881434))",
              ("koeln_planet_osm_line_lines", "koeln_planet_osm_polygon_polygons")),
    "leipzig": ("POLYGON ((12.259321149958366 51.39186504610791,"
                " 12.259321149958366 51.25261834150595,"
                " 12.508230146540393 51.25261834150595,"
                " 12.508230146540393 51.39186504610791,"
                " 12.259321149958366 51.39186504610791))",
                ("leipzig_planet_osm_line_lines", "leipzig_planet_osm_polygon_polygons")),
    "munic": ("POLYGON ((11.80283499714479 48.27244162486505,"
              " 11.80283499714479 48.004382910283,"
              " 11.344842443433851 48.004382910283,"
              " 11.344842443433851 48.27244162486505,"
              " 11.80283499714479 48.27244162486505))",
              ("munic_planet_osm_line_lines", "munic_planet_osm_polygon_polygons")),
    "nuernberg": ("POLYGON ((10.940590168428631 49.49230171195612,"
                  " 10.940590168428631 49.38916478298833,"
                  " 11.160316730928631 49.38916478298833,"
                  " 10.940590168428631 49.49230171195612))",
                  ("nuernberg_planet_osm_line_lines", "nuernberg_planet_osm_polygon_polygons")),
    "stuttgart": ("POLYGON ((9.090103895807271 48.827549744661496,"
                  " 9.090103895807271 48.701952248207306,"
                  " 9.339699537897115 48.701952248207306,"
                  " 9.339699537897115 48.827549744661496,"
                  " 9.090103895807271 48.827549744661496))",
                  ("stuttgart_planet_osm_line_lines", "stuttgart_planet_osm_polygon_polygons")),
    "istanbul": ("POLYGON ((28.46385498046875 41.41175476637062,"
                 " 28.46385498046875 40.701381759859544,"
                 " 29.809680175781253 40.701381759859544,"
                 " 29.809680175781253 41.41175476637062,"
                 " 28.46385498046875 41.41175476637062 ))",
                 ("istanbul_planet_osm_line_lines", "istanbul_planet_osm_polygon_polygons")),
    "moskau": ("POLYGON ((36.86 55.329999999999984,"
               " 36.86 56.170000000000016,"
               " 38.37 56.170000000000016,"
               " 38.37 55.329999999999984,"
               " 36.86 55.329999999999984 ))",
               ("moskau_planet_osm_line_lines", "moskau_planet_osm_polygon_polygons")),
    "london": ("POLYGON  ((-0.5 51.30000000000001,"
               " -0.5 51.68000000000001,"
               " 0.28 51.68000000000001,"
               " 0.28 51.30000000000001,"
               " -0.5 51.30000000000001 ))",
               ("london_planet_osm_line_lines", "london_planet_osm_polygon_polygons")),
    "st_petersburg": ("POLYGON ((29.73 59.650000000000006,"
                      " 29.73 60.22999999999999,"
                      " 30.899999999999995 60.22999999999999,"
                      " 30.899999999999995 59.650000000000006,"
                      " 29.73 59.650000000000006 ))",
                      ("st_petersburg_planet_osm_line_lines", "st_petersburg_planet_osm_polygon_polygons")),
    "madrid": ("POLYGON ((-4.01 40.180000000000035,"
               " -4.01 40.650000000000034,"
               " -3.39 40.650000000000034,"
               " -3.39 40.180000000000035,"
               " -4.01 40.180000000000035 ))",
               ("madrid_planet_osm_line_lines", "madrid_planet_osm_polygon_polygons")),
    "kiew": ("POLYGON ((30.18 50.24000000000001,"
             " 30.18 50.66999999999999,"
             " 30.87 50.66999999999999,"
             " 30.87 50.24000000000001,"
             " 30.18 50.24000000000001 ))",
             ("kiew_planet_osm_line_lines", "kiew_planet_osm_polygon_polygons")),
    "rom": ("POLYGON ((12.329999999999998 41.76999999999998,"
            " 12.329999999999998 42.05000000000001,"
            " 12.73 42.05000000000001,"
            " 12.73 41.76999999999998,"
            " 12.329999999999998 41.76999999999998 ))",
            ("rom_planet_osm_line_lines", "rom_planet_osm_polygon_polygons")),
    "paris": ("POLYGON ((2.158899046092055 48.956146804198454,"
              " 2.158899046092055 48.694405931857915,"
              " 2.6601502667951804 48.694405931857915,"
              " 2.6601502667951804 48.956146804198454,"
              " 2.158899046092055 48.956146804198454 ))",
              ("paris_planet_osm_line_lines", "paris_planet_osm_polygon_polygons")),
    "minsk": ("POLYGON ((27.27 53.72999999999999,"
              " 27.27 54.06999999999999,"
              " 27.86 54.06999999999999,"
              " 27.86 53.72999999999999,"
              " 27.27 53.72999999999999 ))",
              ("minks_planet_osm_line_lines", "minks_planet_osm_polygon_polygons")),  # todo rename table to minsk
    "wien": ("POLYGON ((15.989999999999998 47.99000000000004,"
             " 15.989999999999998 48.360000000000014,"
             " 16.67 48.360000000000014,"
             " 16.67 47.99000000000004,"
             " 15.989999999999998 47.99000000000004 ))",
             ("wien_planet_osm_line_lines", "wien_planet_osm_polygon_polygons")),
    "bukarest": ("POLYGON ((25.860000000000003 44.25,"
                 " 25.860000000000003 44.610000000000014,"
                 " 26.36 44.610000000000014,"
                 " 26.36 44.25,"
                 " 25.860000000000003 44.25 ))",
                 ("bukarest_planet_osm_line_lines", "bukarest_planet_osm_polygon_polygons")),
    "budapest": ("POLYGON ((18.79 47.329999999999984,"
                 " 18.79 47.670000000000016,"
                 " 19.29 47.670000000000016,"
                 " 19.29 47.329999999999984,"
                 " 18.79 47.329999999999984 ))",
                 ("budapest_planet_osm_line_lines", "budapest_planet_osm_polygon_polygons")),
    "warschau": ("POLYGON ((20.729999999999997 52.06,"
                 " 20.729999999999997 52.39999999999998,"
                 " 21.289999999999996 52.39999999999998,"
                 " 21.289999999999996 52.06,"
                 " 20.729999999999997 52.06 ))",
                 ("warschau_planet_osm_line_lines", "warschau_planet_osm_polygon_polygons")),
    "barcelona": ("POLYGON ((1.94 41.21999999999997,"
                  " 1.94 41.55000000000001,"
                  " 2.38 41.55000000000001,"
                  " 2.38 41.21999999999997,"
                  " 1.94 41.21999999999997 ))",
                  ("barcelona_planet_osm_line_lines", "barcelona_planet_osm_polygon_polygons")),
    "charkiw": ("POLYGON ((35.96380823159821 50.17436973639889,"
                " 35.96380823159821 49.673960463444615,"
                " 36.77542322183258 49.673960463444615,"
                " 36.77542322183258 50.17436973639889,"
                " 35.96380823159821 50.17436973639889 ))",
                ("charkiw_planet_osm_line_lines", "charkiw_planet_osm_polygon_polygons")),
    "mailand": ("POLYGON ((8.929009246826167 45.60914909045712,"
                " 8.929009246826167 45.222591504600985,"
                " 9.584069061279292 45.222591504600985,"
                " 9.584069061279292 45.60914909045712,"
                " 8.929009246826167 45.60914909045712 ))",
                ("mailand_planet_osm_line_lines", "mailand_planet_osm_polygon_polygons")),
    "prag": ("POLYGON ((14.152312482024763 50.211939636166534,"
             " 14.152312482024763 49.906008080639,"
             " 14.788146222259138 49.906008080639,"
             " 14.788146222259138 50.211939636166534,"
             " 14.152312482024763 50.211939636166534 ))",
             ("prag_planet_osm_line_lines", "prag_planet_osm_polygon_polygons")),
    "nischni_nowgorod": ("POLYGON (( 43.74 56.18000000000001,"
                         " 43.74 56.47999999999999,"
                         " 44.27000000000001 56.47999999999999,"
                         " 44.27000000000001 56.18000000000001,"
                         " 43.74 56.18000000000001))",
                         ("nischni_nowgorod_planet_osm_line_lines", "nischni_nowgorod_planet_osm_polygon_polygons")),
    "sofia": ("POLYGON ((23.13 42.56,"
              " 23.13 42.84,"
              " 23.51 42.84,"
              " 23.51 42.56,"
              " 23.13 42.56 ))",
              ("sofia_planet_osm_line_lines", "sofia_planet_osm_polygon_polygons")),
    "belgrad": ("POLYGON ((20.070522155761726 45.04836293582332,"
                " 20.070522155761726 44.52105286856175,"
                " 20.84231170654298 44.52105286856175,"
                " 20.84231170654298 45.04836293582332,"
                " 20.070522155761726 45.04836293582332 ))",
                ("belgrad_planet_osm_line_lines", "belgrad_planet_osm_polygon_polygons")),
    "kasan": ("POLYGON ((48.727088928222635 55.92483305458492,"
              " 48.727088928222635 55.57937040601141,"
              " 49.45081329345701 55.57937040601141,"
              " 49.45081329345701 55.92483305458492,"
              " 48.727088928222635 55.92483305458492 ))",
              ("kasan_planet_osm_line_lines", "kasan_planet_osm_polygon_polygons")),
    "samara": ("POLYGON ((49.92000000000001 53.06,"
               " 49.92000000000001 53.34,"
               " 50.38 53.34,"
               " 50.38 53.06,"
               " 49.92000000000001 53.06 ))",
               ("samara_planet_osm_line_lines", "samara_planet_osm_polygon_polygons")),
    "rostow": ("POLYGON ((39.44225189208985 47.313075365595466,"
               " 39.44225189208985 47.03488489003283,"
               " 40.001181335449225 47.03488489003283,"
               " 40.001181335449225 47.313075365595466,"
               " 39.44225189208985 47.313075365595466 ))",
               ("rostow_planet_osm_line_lines", "rostow_planet_osm_polygon_polygons")),
    "birmingham": ("POLYGON ((-2.1662875366210943 52.64720918541238,"
                   " -2.1662875366210943 52.29502164934905,"
                   " -1.5689059448242193 52.29502164934905,"
                   " -1.5689059448242193 52.64720918541238,"
                   " -2.1662875366210943 52.64720918541238 ))",
                   ("birmingham_planet_osm_line_lines", "birmingham_planet_osm_polygon_polygons")),
    "ufa": ("POLYGON ((55.74 54.610000000000014,"
            " 55.74 54.879999999999995,"
            " 56.2 54.879999999999995,"
            " 56.2 54.610000000000014,"
            " 55.74 54.610000000000014 ))",
            ("ufa_planet_osm_line_lines", "ufa_planet_osm_polygon_polygons"))
}

feature_ids = {
    "unlabelled": 0,
    "building": 1,
    "wood": 2,
    "water": 3,
    "road": 4,
    "residential": 5
}
# Not used
feature_id_to_color = {
    0: (0, 0, 0),
    1: (213, 131, 7),
    2: (0, 153, 0),
    3: (0, 0, 204),
    4: (76, 0, 153),
    5: (255, 255, 102)
}
# Not used
color_to_feature_id = {
    (0, 0, 0): 0,
    (213, 131, 7): 1,
    (0, 153, 0): 2,
    (0, 0, 204): 3,
    (76, 0, 153): 4,
    (255, 255, 102): 5
}


def geojson_to_wkt(geojsonfile):
    from shapely.geometry import shape
    import geojson

    with open(geojsonfile) as f:
        gj = geojson.load(f)
    geom = shape(gj)
    return geom.wkt


def get_query_string(type, poly_bbox, region="unterfranken"):
    """

    :param type: a string defining the kind of feature polygons to be queried
    :param area: the area serving as source for the geo data, this is a key used for the dict
                 table_names
    :param poly_bbox: the rectangular space used to intersect the found polygons
    :return: a postgis query string for getting a list containing the not-null intersections of
     all found polygons of type type with poly_bbox
     Thx to https://stackoverflow.com/a/29831823/8862202 (SUM OVER ())
    """
    type = type.lower()
    table_lines, table_polys = table_names[region][1]
    road_types = {
        "primary": 0.000045,
        "secondary": 0.00003,
        "tertiary": 0.000025,
        "residential": 0.00003,
        "service": 0.000025,
        "track": 0.00003,
        "road": 0.00003,
        "unclassified": 0.00003,
        "pedestrian": 0.000015,
        "footway": 0.000015,
        "path": 0.000015
    }
    road_default_width = 0.00003
    if type == "building":
        query = "SELECT ST_AsText(xyz.geo) AS polybuilding  " \
                "FROM ( " \
                "SELECT building, ST_Intersection('" + poly_bbox.wkt + "',geom) AS geo " \
                                                                       "FROM " + table_polys + " " \
                                                                                               "WHERE building IS NOT NULL AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom)) AS xyz " \
                                                                                                                                                                                       "" \
                                                                                                                                                                                       "; "
    elif type == "road":
        query = "SELECT ST_AsText(xyz.geo) AS polyroads " \
                "FROM ( "
        road_types_list = list(road_types.keys())
        helper_default = ""
        for type in road_types_list:
            helper_default += " AND highway!='" + type + "' "
            query += "SELECT ST_Intersection('" + poly_bbox.wkt + "',ST_Buffer(geom, " + str(road_types[type]) + ", " \
                                                                                                                 "'endcap=square join=round')) AS geo " \
                                                                                                                 "FROM " + table_lines + " " \
                                                                                                                                         "WHERE ( highway ='" + type + "' ) " \
                                                                                                                                                                       "AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom) "
            if type != road_types_list[-1]:
                query += "UNION "
        # default road
        query += " UNION SELECT ST_Intersection('" + poly_bbox.wkt + "',ST_Buffer(geom, " + str(
            road_default_width) + ", " \
                                  "'endcap=square join=round')) AS geo " \
                                  "FROM " + table_lines + " " \
                                                          "WHERE (highway is not null " + helper_default + " ) " \
                                                                                                           "AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom) "

        query += ") AS xyz " \
                 "; "
        # TODO get roads buffer width...lookup-table?
    elif type == "wood":
        query = "SELECT ST_AsText(xyz.geo) AS polyforest " \
                "FROM ( " \
                "SELECT ST_Intersection('" + poly_bbox.wkt + "',geom) AS geo " \
                                                             "FROM " + table_polys + " " \
                                                                                     "WHERE ( 'natural'='wood' OR landuse='forest' OR leisure='park') " \
                                                                                     "AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom)) AS xyz " \
                                                                                                                                                  "; "
    elif type == "water":
        # Todo Add Lakes, pools etc., river-plotting fix, river-problem: no width
        query = "SELECT ST_AsText(xyz.geo) " \
                "FROM ( " \
                "SELECT ST_Intersection('" + poly_bbox.wkt + "'," \
                                                             "ST_Buffer(geom, to_number(width, '999D9')* 0.000005, 'endcap=round join=round')) AS geo " \
                                                             "FROM " + table_lines + " " \
                                                                                     "WHERE ( waterway='river' OR waterway='drain' ) " \
                                                                                     "AND ST_Intersects('" + poly_bbox.wkt + "',geom) " \
                                                                                                                             "UNION " \
                                                                                                                             "SELECT ST_Intersection('" + poly_bbox.wkt + "',geom) AS geo " \
                                                                                                                                                                          "FROM " + table_polys + " " \
                                                                                                                                                                                                  "WHERE ( 'natural'='water' OR 'natural'='lake' OR water IS NOT NULL OR leisure='swimming_pool' " \
                                                                                                                                                                                                  "OR leisure='swimming_area') " \
                                                                                                                                                                                                  "AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom) " \
                                                                                                                                                                                                                                                               ")AS xyz " \
                                                                                                                                                                                                                                                               "; "
    elif type == "residential":
        query = "SELECT ST_AsText(xyz.geo) AS polyresident " \
                "FROM ( " \
                "SELECT ST_Intersection('" + poly_bbox.wkt + "',geom) AS geo " \
                                                             "FROM " + table_polys + " " \
                                                                                     "WHERE (  landuse='residential') " \
                                                                                     "AND ST_IsValid(geom) AND ST_Intersects('" + poly_bbox.wkt + "',geom)) AS xyz " \
                                                                                                                                                  "; "
    # "SELECT ST_Intersection('" + poly_bbox.wkt + "',ST_Buffer(geom, to_number(width, '999D9')/10000)) AS geo "\
    return query


class DefaultConfig(object):
    """
    Default Configuration for image loading
    """
    # IMPORTANT: Max Height/Width for google static maps = 640x640!
    pxwidth = 600
    pxheight = 620
    zoomfaktor = 17
    # google api key
    API_KEY = 'AIzaSyDGB5AkLLgzXB6_rUwVjWIUqQ1FT3sXzO0'
    offset = 20
    use_crop_offset = True  # Todo if yes cut offset when loading if false do not
    # the global data directory to store the google images
    datadir = '../ANN_DATA'
    # the number of images to be loaded (= epoch_size of the dataset to be constructed)
    num_images = 7000
    custom_dir = None
    custom_dir_gt = None
    # if False data shall be loaded from previously stored osm files instead of db
    # if true data will be rendered live from db data
    load_from_db = False
    image_index = 0  # current image_index, used by get_sample()

    # custom_dir = 'C:\Uni\Masterstudium\ma-werthmann\code\ANN_DATA\GoogleImages_16_224x244'

    def get_complete_dir(self):
        """
        Returns the full directory path
        :return: the data directory to store the google images
        """
        if self.custom_dir is not None:
            return self.custom_dir
        return self.datadir + '/GoogleImages_' + str(self.zoomfaktor) + '_' + str(self.pxwidth) + 'x' + str(
            self.pxheight)

    def get_gt_dir(self):
        """
        Returns the full directory path to the osm files.
        This directory contains osm files if save_dataset_to_files was invoked before
        :return:
        """
        if self.custom_dir_gt is not None:
            return self.custom_dir_gt
        return self.datadir + '/GoogleImages_' + str(self.zoomfaktor) + '_' + str(self.pxwidth) + 'x' + str(
            self.pxheight) + '/gt_data'


def convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat, maxwidth, maxheight, cutToDigits=11):
    """
    :param cutToDigits determines how to how many digits the coords shall be cut
    :return:
    """
    lon = float(format(lon, '.' + str(cutToDigits) + 'f'))
    lat = float(format(lat, '.' + str(cutToDigits) + 'f'))
    lonmax = float(format(lonmax, '.' + str(cutToDigits) + 'f'))
    latmax = float(format(latmax, '.' + str(cutToDigits) + 'f'))
    lonmin = float(format(lonmin, '.' + str(cutToDigits) + 'f'))
    latmin = float(format(latmin, '.' + str(cutToDigits) + 'f'))

    if lon < lonmin:
        lon = lonmin
    if lon > lonmax:
        lon = lonmax
        # print('lon out of bounds (was '+str(lon)+', bounds: '+str(lonmin)+' - '+str(lonmax)+')')
    if lat < latmin:
        lat = latmin
    if lat > latmax:
        lat = latmax
        # print('lat out of bounds (was '+str(lat)+', bounds: '+str(latmin)+' - '+str(latmax)+')')
    resx = int(((lon - lonmin) / (lonmax - lonmin)) * maxwidth)
    resy = int(((lat - latmin) / (latmax - latmin)) * maxheight)
    return resx, resy


def getRandomLatLongIn(BBox_WKT, meter_offset=200):
    if not BBox_WKT:
        poly = loads(
            "POLYGON((9.086199142965881 50.07247606571221,"
            " 9.086199142965881 49.66887056979786,"
            " 10.61879191640338 49.66887056979786,"
            " 10.61879191640338 50.07247606571221,"
            " 9.086199142965881 50.07247606571221))")
    else:
        poly = loads(BBox_WKT)
    bbox = poly.bounds

    # shrink bbox by meter_offset to avoid missing labels for cut image
    # 224x224 px ~ 355.35 meter (latdiff = 0,003194705221269) x 326.23 meter (londiff = 0,004806518553899)
    # ~ latdif per meter: 0,00000899030595544 londiff per meter: 0,00001473352712472488
    lat_offset = 0.00000899030595544 * 200
    lon_offset = 0.00001473352712472488 * 200
    minlon, maxlon = bbox[0] + lon_offset, bbox[2] - lon_offset
    minlat, maxlat = bbox[1] + lat_offset, bbox[3] - lat_offset

    # roll random number inside of bbox
    lon = random.uniform(minlon, maxlon)
    lat = random.uniform(minlat, maxlat)
    return lat, lon


def getPolygonsForFeatures(lat, lon, feature, config=DefaultConfig(), region="unterfranken"):
    """
    Obtains the desired feature polygons from the postgis db

    :param lat: the lat coord part (which has been used to obtain the google image)
    :param lon: the lon coord part (which has been used to obtain the google image)
    :param feature: can be either "road", "building", "wood" or "water"
    :param config: see DefaultConfig
    :param region: A string specifying a key used to get the table for the chosen region of data
    :return: a tuple cointing 1)all polygons for features which are part of the image of interest
                              2)the percentage share of the area of the desired feature
    """
    # get config values
    zoom = config.zoomfaktor
    pxwidth = config.pxwidth
    pxheight = config.pxheight

    # Get Env
    with open(config.datadir + '/ENV') as env_file:
        env = env_file.readline().lower()

    # Get Connection String
    con_string = connect_strings[env]

    # Connect to db
    conn = pg.connect(con_string)

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # use random coords as center for google img and calculate longlat bounds of google images
    centerPoint = helper.G_LatLng(lat, lon)
    corners = helper.getCorners(centerPoint, zoom, pxwidth, pxheight)
    bbox = corners['W'], corners['S'], corners['E'], corners['N']
    bbox_w, bbox_s, bbox_e, bbox_n = bbox
    lonmin, latmin, lonmax, latmax = bbox
    ext = [(bbox_w, bbox_s),
           (bbox_w, bbox_n),
           (bbox_e, bbox_n),
           (bbox_e, bbox_s),
           (bbox_w, bbox_s)]

    poly_bbox = Polygon(ext)
    q_string = get_query_string(feature, poly_bbox, region=region)
    cur.execute(q_string)

    # get query results
    res = cur.fetchall()

    # transform longlat-coords into XY-Pixel-Coordinates
    polys_features = []
    area = 0
    for row in res:
        if row[0] is None:
            continue
        polygon = loads(row[0])
        if isinstance(polygon, shapely.geometry.polygon.Polygon):
            poly = []
            for lon, lat in polygon.exterior.coords[:]:
                x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                pxwidth, pxheight)
                poly.append((x, y))
            polys_features.append(Polygon(poly))
        elif isinstance(polygon, shapely.geometry.multipolygon.MultiPolygon):
            for sub_poly in polygon:
                poly = []
                for lon, lat in sub_poly.exterior.coords[:]:
                    x, y = convert_longLat_to_pixel(latmin, lonmin, latmax, lonmax, lon, lat,
                                                    pxwidth, pxheight)
                    poly.append((x, y))
                polys_features.append(Polygon(poly))
        else:
            print(feature + ': ' + str(polygon.__class__()))
    # Close communication with the database
    cur.close()
    conn.close()

    return polys_features, area


# Make the changes to the database persistent
# conn.commit()
def initialize(data_folder):
    # 1) get all png files in data folder and save into list
    old_dir = os.getcwd()
    os.chdir(data_folder)
    for file in glob.glob("*.png"):
        png_files.append(file)
    # 2) set init flag to false
    global initialized
    initialized = True
    os.chdir(old_dir)


def maybe_download_images(config=DefaultConfig(), img_regions=None):
    """

    :param config: a config object
    :param img_regions: a list containing keys of Map tablenames
    determining the tables/areas the images will be loaded from. If region
    contains tuples (region1, 1000), (region2, 1500), this method will ignore
    config.num_labels and load as many images as described in the tuples respectively from each region
    :return:
    """
    dirname = config.get_complete_dir()
    if not os.path.isdir(config.datadir):
        os.mkdir(config.datadir)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

        glob_feature_distrib = {
            "wood": 0,
            "building": 0,
            "water": 0,
            "road": 0,
            "unlabelled": 0,
            "residential": 0
        }
        if img_regions is not None:
            regions = list(img_regions)
        else:
            regions = list(table_names.keys())
        # Check if list of form [('area1', 500),('area2', 300)...] was given
        #  In this case config.num_images will be ignored
        total_num_images = config.num_images
        if len(regions[0]) == 2:
            # for each region
            print('nice richtig')
            total_num_images = 0
            for r in regions:
                region_name, num_images = r[0], r[1]
                for i in range(0, num_images):
                    region = region_name
                    bigbbox = table_names[region][0]
                    while True:
                        try:
                            r_lat, r_lon = getRandomLatLongIn(bigbbox)
                            myconfig = DefaultConfig()
                            polys = dict()
                            polys["wood"] = getPolygonsForFeatures(r_lat, r_lon, feature="wood", config=myconfig,
                                                                   region=region)
                            polys["building"] = getPolygonsForFeatures(r_lat, r_lon, feature="building",
                                                                       config=myconfig,
                                                                       region=region)
                            polys["water"] = getPolygonsForFeatures(r_lat, r_lon, feature="water", config=myconfig,
                                                                    region=region)
                            polys["road"] = getPolygonsForFeatures(r_lat, r_lon, feature="road", config=myconfig,
                                                                   region=region)
                            polys["residential"] = getPolygonsForFeatures(r_lat, r_lon, feature="residential",
                                                                          config=myconfig,
                                                                          region=region)
                            train_mask = colorPolys(polygons=polys["wood"][0],
                                                    poly_type=feature_ids["wood"],
                                                    im_size=(myconfig.pxwidth, myconfig.pxheight))
                            for feature in ["water", "residential", "building", "road"]:
                                train_mask = colorPolys(polygons=polys[feature][0],
                                                        poly_type=feature_ids[feature],
                                                        im_size=(myconfig.pxwidth, myconfig.pxheight),
                                                        img_mask=train_mask)

                            feature_distrib = evaluate_feature_distrib(train_mask)
                            # print(feature_distrib)

                            if not is_valid_image(img_feature_dict=feature_distrib):
                                # don't load google image and roll next coords
                                continue
                            else:
                                # if valid -> add feature distrib to sum & exit endless loop and calc next image
                                glob_feature_distrib["wood"] = glob_feature_distrib["wood"] \
                                                               + feature_distrib[feature_ids["wood"]]

                                glob_feature_distrib["building"] = glob_feature_distrib["building"] \
                                                                   + feature_distrib[feature_ids["building"]]

                                glob_feature_distrib["water"] = glob_feature_distrib["water"] \
                                                                + feature_distrib[feature_ids["water"]]

                                glob_feature_distrib["road"] = glob_feature_distrib["road"] \
                                                               + feature_distrib[feature_ids["road"]]

                                glob_feature_distrib["residential"] = glob_feature_distrib["residential"] \
                                                                      + feature_distrib[feature_ids["residential"]]

                                glob_feature_distrib["unlabelled"] = glob_feature_distrib["unlabelled"] \
                                                                     + feature_distrib[feature_ids["unlabelled"]]
                                break
                        except TypeError:  # most of the errors seem to be caused by water polygons
                            print('Type Error')
                            print(polys["water"])

                    # valid image found -> load corresponding google image
                    url = 'http://maps.googleapis.com/maps/api/staticmap' + '?' \
                          + 'key=' + config.API_KEY \
                          + '&center=' + str(r_lat) + ',' \
                          + str(r_lon) + \
                          '&size=' + str(config.pxwidth) + 'x' + str(config.pxheight) + \
                          '&zoom=' + str(config.zoomfaktor) + \
                          '&sensor=false&maptype=satellite&style=feature:all|element:labels|visibility:off'
                    buffer = BytesIO(request.urlopen(url).read())
                    image = Image.open(buffer)
                    file_name = region + '_' + str(r_lat) + 'x' + str(r_lon) + '.png'
                    image.save(dirname + '/' + file_name)
                    print(file_name + ' loaded: ' + str(i + 1))

        else:
            for i in range(0, config.num_images):
                region = regions[i % len(regions)]
                bigbbox = table_names[region][0]
                while True:
                    try:
                        r_lat, r_lon = getRandomLatLongIn(bigbbox)
                        myconfig = DefaultConfig()
                        polys = dict()
                        polys["wood"] = getPolygonsForFeatures(r_lat, r_lon, feature="wood", config=myconfig,
                                                               region=region)
                        polys["building"] = getPolygonsForFeatures(r_lat, r_lon, feature="building", config=myconfig,
                                                                   region=region)
                        polys["water"] = getPolygonsForFeatures(r_lat, r_lon, feature="water", config=myconfig,
                                                                region=region)
                        polys["road"] = getPolygonsForFeatures(r_lat, r_lon, feature="road", config=myconfig,
                                                               region=region)
                        polys["residential"] = getPolygonsForFeatures(r_lat, r_lon, feature="residential",
                                                                      config=myconfig,
                                                                      region=region)
                        train_mask = colorPolys(polygons=polys["wood"][0],
                                                poly_type=feature_ids["wood"],
                                                im_size=(myconfig.pxwidth, myconfig.pxheight))
                        for feature in ["water", "residential", "building", "road"]:
                            train_mask = colorPolys(polygons=polys[feature][0],
                                                    poly_type=feature_ids[feature],
                                                    im_size=(myconfig.pxwidth, myconfig.pxheight),
                                                    img_mask=train_mask)

                        feature_distrib = evaluate_feature_distrib(train_mask)
                        print(feature_distrib)

                        if not is_valid_image(img_feature_dict=feature_distrib):
                            # don't load google image and roll next coords
                            continue
                        else:
                            # if valid -> add feature distrib to sum & exit endless loop and calc next image
                            glob_feature_distrib["wood"] = glob_feature_distrib["wood"] \
                                                           + feature_distrib[feature_ids["wood"]]

                            glob_feature_distrib["building"] = glob_feature_distrib["building"] \
                                                               + feature_distrib[feature_ids["building"]]

                            glob_feature_distrib["water"] = glob_feature_distrib["water"] \
                                                            + feature_distrib[feature_ids["water"]]

                            glob_feature_distrib["road"] = glob_feature_distrib["road"] \
                                                           + feature_distrib[feature_ids["road"]]

                            glob_feature_distrib["residential"] = glob_feature_distrib["residential"] \
                                                                  + feature_distrib[feature_ids["residential"]]

                            glob_feature_distrib["unlabelled"] = glob_feature_distrib["unlabelled"] \
                                                                 + feature_distrib[feature_ids["unlabelled"]]
                            break
                    except TypeError:  # most of the errors seem to be caused by water polygons
                        print('Type Error')
                        print(polys["water"])

                # valid image found -> load corresponding google image
                url = 'http://maps.googleapis.com/maps/api/staticmap' + '?' \
                      + 'key=' + config.API_KEY \
                      + '&center=' + str(r_lat) + ',' \
                      + str(r_lon) + \
                      '&size=' + str(config.pxwidth) + 'x' + str(config.pxheight) + \
                      '&zoom=' + str(config.zoomfaktor) + \
                      '&sensor=false&maptype=satellite&style=feature:all|element:labels|visibility:off'
                buffer = BytesIO(request.urlopen(url).read())
                image = Image.open(buffer)
                file_name = region + '_' + str(r_lat) + 'x' + str(r_lon) + '.png'
                image.save(dirname + '/' + file_name)
                print(file_name + ' loaded: ' + str(i + 1))

        # after loop calc average featur distrib
        glob_feature_distrib["wood"] = round(100 * glob_feature_distrib["wood"] / total_num_images, 2)
        glob_feature_distrib["building"] = round(100 * glob_feature_distrib["building"] / total_num_images, 2)
        glob_feature_distrib["water"] = round(100 * glob_feature_distrib["water"] / total_num_images, 2)
        glob_feature_distrib["road"] = round(100 * glob_feature_distrib["road"] / total_num_images, 2)
        glob_feature_distrib["residential"] = round(100 * glob_feature_distrib["residential"] / total_num_images, 2)
        glob_feature_distrib["unlabelled"] = round(100 * glob_feature_distrib["unlabelled"] / total_num_images, 2)
        # save config and feature distrib average
        # thx 2 https://stackoverflow.com/a/36965528/8862202
        with open(config.get_complete_dir() + '/feature_distrib.txt', 'w') as file:
            file.write(json.dumps(glob_feature_distrib))  # use `json.loads` to do the reverse
        with open(config.get_complete_dir() + '/config.txt', 'w') as file:
            for attr in dir(config):
                if not attr.startswith("__"):
                    file.write('%s = %r\n' % (attr, getattr(config, attr)))  # use `json.loads` to do the reverse


def split_dataset(config=DefaultConfig(), train=0.8, test=0.2, val=0, to_folder=None, ds_name='osm'):
    """
    Splits the dataset into train and test dataset and optionally into val.
    Requires the dataset to be represent as files (use save_dataset_to_files() before)

    :param config:
    :param train: the percentage of the dataset to build the training data
    :param test: the percentage of the dataset to build the test data
    :param val: the percentage of the dataset to build the validation data
    :param to_file: the folder for the list to be stored [optional]
    :param ds_name: the name used to name the generated files. Will be ignored if to_folder is None [optional]
    :return: (train_file_names, test_file_names) or (train_file_names, val_file_names test_file_names) if
              val was greater 0
    """
    sum = train + test + val
    if sum != 1:
        raise ValueError('sum of train, test and val has to be 1 (was %f)!' % (3))
    if val >= train:
        raise ValueError('val must be smaller than train!')

    # Load dataset file names
    old_dir = os.getcwd()
    os.chdir(config.get_complete_dir())
    file_names = []
    for file in glob.glob("*.png"):
        file_names.append(file)
    os.chdir(old_dir)

    np.random.shuffle(file_names)
    dataset_size = len(file_names)
    # Split into test and (val+train)
    test_idx = round((train + val) * dataset_size)
    print("dataset size: "+str(dataset_size))
    print("testindex: "+str(test_idx))

    train_val_data, test_data = file_names[:test_idx], file_names[test_idx:]
    if val == 0:# Split into TRAIN|TEST
        train_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in train_val_data]
        test_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in test_data]
        dataset_split = train_data_files, test_data_files
    else: # Split into VAL|TRAIN|TEST
        val_idx = round(val * dataset_size)
        print("valindex: " + str(val_idx))
        val_data, train_data = train_val_data[:val_idx], train_val_data[val_idx:test_idx]
        train_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in train_data]
        val_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in val_data]
        test_data_files = [(f, 'gt_data/' + 'GT_' + f) for f in test_data]
        dataset_split = train_data_files, val_data_files, test_data_files

    # write train file
    if to_folder is not None and os.path.isdir(to_folder):
        with open(to_folder + '/' + ds_name + '_train_list.txt', 'wt') as f:
            for line in train_data_files:
                x, y = line
                f.write(x + ' ' + y + '\n')
            f.flush()
    # write test file
    if to_folder is not None and os.path.isdir(to_folder):
        with open(to_folder + '/' + ds_name + '_test_list.txt', 'wt') as f:
            for line in test_data_files:
                x, y = line
                f.write(x + ' ' + y + '\n')
            f.flush()
    # write val file
    if val > 0:
        if to_folder is not None and os.path.isdir(to_folder):
            with open(to_folder + '/' + ds_name + '_val_list.txt', 'wt') as f:
                for line in val_data_files:
                    x, y = line
                    f.write(x + ' ' + y + '\n')
                f.flush()
    return dataset_split


def crop_offset_from_dataset(config=DefaultConfig()):
    if not initialized:
        initialize(config.get_complete_dir())
    gt_dir = config.get_gt_dir()
    complete_dir_new = config.get_complete_dir() + "_cropped"
    gt_dir_new = complete_dir_new + "/gt_data"
    if not os.path.isdir(gt_dir):
        os.mkdir(gt_dir)
    if not os.path.isdir(complete_dir_new):
        os.mkdir(complete_dir_new)
    if not os.path.isdir(gt_dir_new):
        os.mkdir(gt_dir_new)
    count = 0
    for idx in trange(len(png_files), desc='crop_offset_from_dataset', leave=True):
        file_name = png_files[idx]
        gt_file_name = gt_dir + '/GT_' + file_name
        # load and crop satellite image
        image = np.asarray(Image.open(config.get_complete_dir() + '/' + file_name).convert('RGB'))
        image = image[:][:-config.offset][:]
        image = Image.fromarray(image)

        # load and crop ground-truth
        train_mask = np.asarray(Image.open(gt_file_name))
        train_mask = train_mask[:][:-config.offset][:]

        train = Image.fromarray(train_mask)
        # save image to file system
        image.save(complete_dir_new + "/" + file_name)
        train.save(gt_dir_new + "/GT_" + file_name)
        count = count + 1
    print(str(count) + " images cropped")


def save_dataset_to_files(config=DefaultConfig()):
    """
    Stores rendered Osm data dataset to files.
    This method loads all data for the osm images from the database,
    labels the images and saves the generated ground-truth image
    to the path given in config
    """
    if not initialized:
        initialize(config.get_complete_dir())
    gt_dir = config.get_gt_dir()
    if not os.path.isdir(gt_dir):
        os.mkdir(gt_dir)
    count = 0
    for idx in trange(len(png_files), desc='Save_dataset_to_files', leave=True):
        file_name = png_files[idx]
        filename_regex = "([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
        m = re.search(filename_regex, file_name)
        # print(file_name)
        region = m.group(1)
        lat = float(m.group(2))
        lon = float(m.group(3))
        if count % (len(png_files) / 10) == 0:
            frac = count / len(png_files)
            print("Saved {0:.3f}% of the dataset !".format(frac * 100) + " (total: " + str(len(png_files)) + ")")
        bigbbox = table_names[region][0]
        polys = dict()
        polys["wood"] = getPolygonsForFeatures(lat, lon, feature="wood", config=config, region=region)
        polys["building"] = getPolygonsForFeatures(lat, lon, feature="building", config=config, region=region)
        polys["water"] = getPolygonsForFeatures(lat, lon, feature="water", config=config, region=region)
        polys["road"] = getPolygonsForFeatures(lat, lon, feature="road", config=config, region=region)
        polys["residential"] = getPolygonsForFeatures(lat, lon, feature="residential", config=config, region=region)
        train_mask = colorPolys(polygons=polys["wood"][0],
                                poly_type=feature_ids["wood"],
                                im_size=(config.pxwidth, config.pxheight))
        for feature in ["water", "residential", "building", "road"]:
            train_mask = colorPolys(polygons=polys[feature][0],
                                    poly_type=feature_ids[feature],
                                    im_size=(config.pxwidth, config.pxheight),
                                    img_mask=train_mask)
        # apply flip
        train_mask = np.flip(train_mask, 0)
        # Note: not cut of pixels here -> has to be done on loading
        im = Image.fromarray(train_mask)
        # save image to file system
        im.save(gt_dir + "/" + "GT_" + file_name, )

        # debug:
        # load image from file system
        # image = np.asarray(
        #     Image.open(gt_dir+"/"+"GT_"+file_name))
        # print(image.shape)
        # plt.imshow(image, vmin=0, vmax=10, cmap='tab10')
        # plt.show()
        count = count + 1


def get_sample(config=DefaultConfig(), get_region=False):
    maybe_download_images(config=config)
    if not initialized:
        initialize(config.get_complete_dir())
    # get file name
    next_image = png_files[config.image_index]
    # extract region, lat, lon from filename
    filename_regex = "([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
    m = re.search(filename_regex, next_image)
    region = m.group(1)
    lat = float(m.group(2))
    lon = float(m.group(3))
    if config.load_from_db:
        bigbbox = table_names[region][0]
        polys = dict()
        polys["wood"] = getPolygonsForFeatures(lat, lon, feature="wood", config=config, region=region)
        polys["building"] = getPolygonsForFeatures(lat, lon, feature="building", config=config, region=region)
        polys["water"] = getPolygonsForFeatures(lat, lon, feature="water", config=config, region=region)
        polys["road"] = getPolygonsForFeatures(lat, lon, feature="road", config=config, region=region)
        polys["residential"] = getPolygonsForFeatures(lat, lon, feature="residential", config=config, region=region)
        train_mask = colorPolys(polygons=polys["wood"][0],
                                poly_type=feature_ids["wood"],
                                im_size=(config.pxwidth, config.pxheight))
        for feature in ["water", "residential", "building", "road"]:
            train_mask = colorPolys(polygons=polys[feature][0],
                                    poly_type=feature_ids[feature],
                                    im_size=(config.pxwidth, config.pxheight),
                                    img_mask=train_mask)

        image = np.asarray(Image.open(config.get_complete_dir() + '/' + next_image).convert('RGB'))
        # plot satellite image (train) and osm labelled image (label)
        train_mask = np.flip(train_mask, 0)
        # Cut to remove google labels at the bottom
        train_mask = train_mask[:][:-config.offset][:]
        image = image[:][:-config.offset][:]
    else:
        # TODO Load data from GT folder and put it into train_mask
        # load gt image from file system
        gt_dir = config.get_gt_dir()
        gt_file_name = gt_dir + "/" + "GT_" + next_image
        if not os.path.isfile(gt_file_name):
            raise FileNotFoundError("Missing ground-truth file! Filename: " + gt_file_name)
        train_mask = np.asarray(
            Image.open(gt_file_name))
        if config.use_crop_offset:
            train_mask = train_mask[:][:-config.offset][:]
        # load satellite image
        image = np.asarray(Image.open(config.get_complete_dir() + '/' + next_image).convert('RGB'))
        if config.use_crop_offset:
            image = image[:][:-config.offset][:]
            # image = np.asarray(Image.open(config.get_complete_dir() + '/' + next_image).convert('RGB'))

    config.image_index = (config.image_index + 1) % config.num_images

    if config.image_index == 0:
        print('Epoch end (' + str(config.num_images) + ') reached')
    if get_region:
        return image, train_mask, region
    else:
        return image, train_mask


def remove_similiar_imgs(data_dir, threshold=0.0045, askValue=0.0045, remove_groundtruth=True):
    """
    This function loops through all images of the given datadir
    and removes images whose coord distance with any other image is lower
    than a threshold. This procedure secures that there are no doubles in the dataset,
    so it can be split into train and test without overlapping.

    If the distance between two images is threshold<dist<askValue
    the distance is and the images are shown for manual comparison

    This function requires the user to manually confirm the number of images to be deleted .
    :return:
    """
    image_names = []
    old_dir = os.getcwd()
    os.chdir(data_dir)
    for file in glob.glob("*.png"):
        image_names.append(file)
    os.chdir(old_dir)

    image_names = sorted(image_names)
    images_to_delete = []
    last_coord = None
    last_img_name = None
    for idx in trange(len(image_names), desc='Evaluate_Feature_Distribution_From_GT', leave=True):
        im_name = image_names[idx]

        filename_regex = "([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
        m = re.search(filename_regex, im_name)
        region = m.group(1)
        lat = float(m.group(2))
        lon = float(m.group(3))
        coord = np.array((lat, lon))
        if last_coord is None:
            last_coord = coord
            last_img_name = im_name
            continue

        # calc distance between this coord and last coord
        dist = np.linalg.norm(coord - last_coord)
        ask = False
        if threshold < dist < askValue:
            print("dist between " + last_img_name + " and " + im_name + " = " + str(dist))
            # plot both to see the diff
            im_new = np.asarray(Image.open(data_dir + '/' + im_name).convert('RGB'))
            im_last = np.asarray(Image.open(data_dir + '/' + last_img_name).convert('RGB'))
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im_new)
            ax[0].set_title(im_name)
            ax[1].imshow(im_last)
            ax[1].set_title(last_img_name)
            plt.show()
            ask = input("Delete this image?(y/n)").lower() == "y"

        if (dist < threshold) or ask:
            images_to_delete.append(im_name)
        last_coord = coord
        last_img_name = im_name

    num_images_del = len(images_to_delete)
    in_line = input(str(num_images_del) + " will be deleted. Do you want to proceed?(y/n)\n")
    if not (in_line.lower() == 'y'):
        return
    # Delete
    for im in images_to_delete:
        os.remove(data_dir + '/'+im)
        if remove_groundtruth:
            os.remove(data_dir + '/gt_data/GT_' + im)



def is_valid_image(img_feature_dict):
    """
    Checks the given feature distribution and returns if this sample shall be used as training data
    :param img_feature_dict: the feature distribution for building, wood, water, road and other

    :return: true if the image is valid, false if invalid (too less or to many of some feature class)
    """
    valid = True
    water = img_feature_dict[feature_ids["water"]]
    building = img_feature_dict[feature_ids["building"]]
    wood = img_feature_dict[feature_ids["wood"]]
    road = img_feature_dict[feature_ids["road"]]
    residential = img_feature_dict[feature_ids["residential"]]
    other = img_feature_dict[feature_ids["unlabelled"]]

    # exclude pure wood
    if wood >= 0.99:
        valid = False
    # exclude pure water
    if water >= 0.99:
        valid = False
    # exclude big unlabelled area
    if other >= 0.40:
        valid = False
    # allow up to 39.9 % of unlabelled pixels if there are buildings
    if other >= 0.30 and building == 0:
        valid = False
    # exlude images with big unlabelled residential areas to improve building detection
    if building <= 0.30 * residential:
        valid = False

    return valid


def evaluate_feature_distrib(image):
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


def evaluate_feature_distrib_from_GT(datadir="../ANN_Data/de_top14_cropped",
                                     img_list="../Segmentation_Models/ICNet/list/de_top14_train_list.txt"):
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


def evaluate_feature_distrib_all(path=None, filter_prefix=None):
    if not path:
        config = DefaultConfig()
        if not initialized:
            initialize(config.get_complete_dir())
    else:
        if not initialized:
            initialize(path)

    total_feature_distrib = {
        "wood": 0,
        "building": 0,
        "water": 0,
        "road": 0,
        "unlabelled": 0,
        "residential": 0
    }
    counter = 0
    for idx in trange(len(png_files), desc='Evaluate_Feature_Distribution_ALL', leave=True):
        image = png_files[idx]
        # check filter
        if filter_prefix is not None and not image.startswith(filter_prefix):
            continue
        counter = counter + 1
        # extract region, lat, lon from filename
        filename_regex = "([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
        m = re.search(filename_regex, image)
        region = m.group(1)
        lat = float(m.group(2))
        lon = float(m.group(3))
        bigbbox = table_names[region][0]
        myconfig = DefaultConfig()
        polys = dict()
        polys["wood"] = getPolygonsForFeatures(lat, lon, feature="wood", config=myconfig, region=region)
        polys["building"] = getPolygonsForFeatures(lat, lon, feature="building", config=myconfig, region=region)
        polys["water"] = getPolygonsForFeatures(lat, lon, feature="water", config=myconfig, region=region)
        polys["road"] = getPolygonsForFeatures(lat, lon, feature="road", config=myconfig, region=region)
        polys["residential"] = getPolygonsForFeatures(lat, lon, feature="residential", config=myconfig, region=region)
        train_mask = colorPolys(polygons=polys["wood"][0],
                                poly_type=feature_ids["wood"],
                                im_size=(myconfig.pxwidth, myconfig.pxheight))
        for feature in ["water", "residential", "building", "road"]:
            train_mask = colorPolys(polygons=polys[feature][0],
                                    poly_type=feature_ids[feature],
                                    im_size=(myconfig.pxwidth, myconfig.pxheight),
                                    img_mask=train_mask)

        # add to sum
        distrib = evaluate_feature_distrib(train_mask)
        for feature_class in total_feature_distrib.keys():
            total_feature_distrib[feature_class] += distrib[feature_ids[feature_class]]

    # calc average
    for feature_class in total_feature_distrib.keys():
        total_feature_distrib[feature_class] /= counter

    if filter_prefix is not None:
        print("Checked " + str(counter) + " filtered images ( of " + str(len(png_files)) + " in total) ")
    return total_feature_distrib


def create_alpha_layer(img, gt_img, num_cls=6):
    # class to color
    label_color = [(0, 0, 0), (81, 0, 81), (128, 64, 128),
                   (244, 35, 232), (250, 170, 160), (230, 150, 140)]
    colored_image = np.zeros(
        (img.shape[0], img.shape[1], 3), np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            colored_image[row, col, :] = label_color[gt_img[row, col]]

    alpha_mask = colored_image
    return alpha_mask


def show_img(img_name, config=DefaultConfig()):
    """
    Displays a sat image, the corresponding GT generated live using PostiGis
    and an alpha layer showing the coverage of the GT on the sat image
    Used for debug only
    :param img_name:
    :param config:
    :return:
    """
    # extract region, lat, lon from filename
    filename_regex = "([\w_]*)_([-0-9]*.[0-9]*)x([-0-9]*.[0-9]*).png"
    m = re.search(filename_regex, img_name)
    region = m.group(1)
    lat = float(m.group(2))
    lon = float(m.group(3))
    bigbbox = table_names[region][0]
    polys = dict()
    polys["wood"] = getPolygonsForFeatures(lat, lon, feature="wood", config=config, region=region)
    polys["building"] = getPolygonsForFeatures(lat, lon, feature="building", config=config, region=region)
    polys["water"] = getPolygonsForFeatures(lat, lon, feature="water", config=config, region=region)
    polys["road"] = getPolygonsForFeatures(lat, lon, feature="road", config=config, region=region)
    polys["residential"] = getPolygonsForFeatures(lat, lon, feature="residential", config=config, region=region)
    train_mask = colorPolys(polygons=polys["wood"][0],
                            poly_type=feature_ids["wood"],
                            im_size=(config.pxwidth, config.pxheight))
    for feature in ["water", "residential", "building", "road"]:
        train_mask = colorPolys(polygons=polys[feature][0],
                                poly_type=feature_ids[feature],
                                im_size=(config.pxwidth, config.pxheight),
                                img_mask=train_mask)

    image = np.asarray(Image.open(config.get_complete_dir() + '/' + img_name).convert('RGB'))
    # plot satellite image (train) and osm labelled image (label)
    train_mask = np.flip(train_mask, 0)
    # Cut to remove google labels at the bottom
    train_mask = train_mask[:][:-config.offset][:]
    image = image[:][:-config.offset][:]

    fig, ax = plt.subplots(1, 3)
    # google image

    ax[0].imshow(image)
    ax[0].set_title('google satellite image')

    # postgis ground-truth image
    ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    ax[1].set_title('postgis-created ground-truth')

    # alpha overlay image
    alpha = create_alpha_layer(image, train_mask)
    ax[2].imshow(image, alpha=0.75)
    ax[2].imshow(alpha, alpha=0.25)
    ax[2].set_title('alpha blend')

    # show max zoomed plot (windwows TkAgg)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()


if __name__ == '__main__':
    # debug compare groundtruth
    # os.chdir("C:/Uni/Masterstudium/ma-werthmann/code/ANN_DATA/GoogleImages_17_600x600_cropped/gt_data")
    # print("OSM GT")
    # for file in glob.glob("*.png"):
    #     gt_img = np.asarray(Image.open(file))
    #     print(gt_img)
    #     print(gt_img.shape)
    #     distrib = evaluate_feature_distrib(gt_img)
    #     print(distrib)
    #     break
    # os.chdir("C:/Uni/Masterstudium/Master Datasets/Cityscapes/gtFine/train/aachen")
    # print("CITYSCAPES GT")
    # for file in glob.glob("*.png"):
    #     if 'label' in file:
    #         pass
    #     else:
    #         continue
    #     print(file)
    #     gt_img = np.asarray(Image.open(file))
    #     print(gt_img)
    #     print(gt_img.shape)
    #     distrib = evaluate_feature_distrib(gt_img)
    #     print(distrib)



    #  plt.figure()



    # lat, lon = 49.7937392,10.1883074 # river test coords
    # lat, lon = 49.7991953, 9.9406978 # wue central test coords
    # lat, lon = 49.79842, 9.92484 # wue river test coords

    # while True:
    #     try:
    #         region = "unterfranken"
    #         bigbbox = table_names[region][0]
    #         lat, lon = getRandomLatLongIn(bigbbox)
    #         myconfig = DefaultConfig()
    #         polys = dict()
    #         polys["wood"] = getPolygonsForFeatures(lat, lon, feature="wood", config=myconfig, region=region )
    #         polys["residential"] = getPolygonsForFeatures(lat, lon, feature="residential", config=myconfig, region=region)
    #         polys["building"] = getPolygonsForFeatures(lat, lon, feature="building", config=myconfig, region=region)
    #         polys["water"] = getPolygonsForFeatures(lat, lon, feature="water", config=myconfig, region=region)
    #         polys["road"] = getPolygonsForFeatures(lat, lon, feature="road", config=myconfig, region=region)
    #         train_mask = colorPolys(polygons=polys["wood"][0],
    #                                 poly_type=feature_ids["wood"],
    #                                 im_size=(myconfig.pxwidth, myconfig.pxheight))
    #         for feature in ["water", "wood", "residential", "building", "road"]:
    #             train_mask = colorPolys(polygons=polys[feature][0],
    #                                     poly_type=feature_ids[feature],
    #                                     im_size=(myconfig.pxwidth, myconfig.pxheight),
    #                                     img_mask=train_mask)
    #
    #         feature_distrib = {
    #             "wood": round(polys["wood"][1]*100, 2),
    #             "building": round(polys["building"][1]*100, 2),
    #             "water": round(polys["water"][1]*100, 2),
    #             "road": round(polys["road"][1]*100, 2),
    #             "other": round(max(0,
    #                                1-polys["wood"][1]-polys["building"][1]-polys["water"][1]-polys["road"][1])*100, 2)
    #         }
    #         print(feature_distrib)
    #         if not is_valid_image(img_feature_dict=feature_distrib):
    #             # don't load google image and roll next coords
    #             continue
    #     except TypeError:
    #         print(polys["water"])
    #         continue
    #     url = 'http://maps.googleapis.com/maps/api/staticmap' + '?' \
    #           + 'key=' + myconfig.API_KEY \
    #           + '&center=' + str(lat) + ',' \
    #           + str(lon) + \
    #           '&size=' + str(myconfig.pxwidth) + 'x' + str(myconfig.pxheight) + \
    #           '&zoom=' + str(myconfig.zoomfaktor) + \
    #           '&sensor=false&maptype=hybrid&style=feature:all|element:labels|visibility:off'
    #     buffer = BytesIO(request.urlopen(url).read())
    #     image = np.asarray(Image.open(buffer).convert('RGB'))
    #     # plot satellite image (train) and osm labelled image (label)
    #     train_mask = np.flip(train_mask, 0)
    #     # Cut to remove google labels at the bottom
    #     train_mask = train_mask[:][:-myconfig.offset][:]
    #     image = image[:][:-myconfig.offset][:]
    #
    #     fig, ax = plt.subplots(1, 2)
    #     # google image
    #
    #     ax[0].imshow(image)
    #     ax[0].set_title('google satellite image')
    #
    #     # postgis ground-truth image
    #     ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    #     ax[1].set_title('postgis-created ground-truth')
    #     plt.show()

    # test calc meters
    # p = loads(table_names["malaga"][0])
    # bbox = p.bounds
    # lat = 52.4983678856
    # lon = 13.3306850560
    # centerPoint = helper.G_LatLng(lat, lon)
    # corners = helper.getCorners(centerPoint, 17, 600, 600)
    # bbox = corners['W'], corners['S'], corners['E'], corners['N']
    # dis1 = helper.getMeterDistance(lon1=bbox[0], lat1=bbox[1], lon2=bbox[0], lat2=bbox[3])
    # dis2 = helper.getMeterDistance(lon1=bbox[0], lat1=bbox[1], lon2=bbox[2], lat2=bbox[1])
    # print(dis1)
    # print(dis2)
    #
    # test feature distrib
    # get images with alpha blend

    # RENDER FULL DATASET AND SAVE GT FILES TO FILE SYSTEM
    # save_dataset_to_files()

    # PLOT 1) SATELLTE,  2) GT AND 3) ALPHA GT
    # show_img('berlin_52.48345219953771x13.564150869852982.png')
    # config = DefaultConfig()
    # Load dataset 'de_top14' - 14 biggest cities in germany
    # config = DefaultConfig()
    # config.num_images = 7000
    # areas = ["berlin", "hamburg", "munic", "koeln", "frankfurt", "stuttgart", "duesseldorf", "dortmund",
    #          "essen", "leipzig", "bremen", "dresden", "hannover", "nuernberg"]
    # maybe_download_images(config, areas)

    # Load dataset 'eu_top25_exde' - 25 biggest cities in europe ex germany
    # config = DefaultConfig()
    # config.custom_dir = '../ANN_DATA/eu_top25_exde'
    # # config.num_images = 7000
    # areas = [("istanbul", 890), ("moskau", 1800), ("london", 1000), ("st_petersburg", 1400), ("madrid", 1000),
    #          ("kiew", 700), ("rom", 450), ("paris", 1000), ("minsk", 400), ("wien", 800), ("bukarest", 400),
    #          ("budapest", 500), ("warschau", 500), ("barcelona", 250), ("charkiw", 700), ("mailand", 700),
    #          ("prag", 500), ("nischni_nowgorod", 400),
    #          ("sofia", 380), ("belgrad", 500), ("kasan", 300), ("samara", 300), ("rostow", 400), ("birmingham", 700),
    #          ("ufa", 200)]
    # maybe_download_images(config, areas)

    # SPLIT 'de_top14' DATASET TO TRAIN, TEST AND VAL
    # config = DefaultConfig()
    # config.custom_dir = 'G:\Datasets\DSTL_Challenge_Dataset'
    # config.custom_dir_gt = 'G:\Datasets\DSTL_Challenge_Dataset\gt_data'
    # split = split_dataset(config, train=0.8, test=0.1, val=0.1, to_folder='.',ds_name='kaggle_dstl')

    # REMOVE images with similiar coords
    eu_set = "H:/FINAL_Datasets/eu_top25_cropped"
    de_set = "H:/FINAL_Datasets/de_top15_cropped"
    tiny_set = "H:/FINAL_Datasets/world_tiny2k_cropped"
    # remove_similiar_imgs(data_dir=tiny_set)
    # remove_similiar_imgs(data_dir=de_set)
    # remove_similiar_imgs(data_dir=eu_set)

    # GET FEATURE DISTRIB FOR DIFFERENT DATASETS
    # img_list_de_train = "../Segmentation_Models/ICNet/list/de_top15_r_train_list.txt"
    # img_list_de_test = "../Segmentation_Models/ICNet/list/de_top15_r_test_list.txt"
    # img_list_de_val = "../Segmentation_Models/ICNet/list/de_top15_r_val_list.txt"
    # fd = evaluate_feature_distrib_from_GT(datadir=de_set, img_list=None)
    # fd_train = evaluate_feature_distrib_from_GT(datadir=de_set, img_list=img_list_de_train)
    # fd_test = evaluate_feature_distrib_from_GT(datadir=de_set, img_list=img_list_de_test)
    # fd_val = evaluate_feature_distrib_from_GT(datadir=de_set, img_list=img_list_de_val)
    # print("de_top14: complete")
    # print(fd)
    # print("de_top14: train")
    # print(fd_train)
    # print("de_top14: test")
    # print(fd_test)
    # print("de_top14: val")
    # print(fd_val)
    # #
    # img_list_eu_train = "../Segmentation_Models/ICNet/list/eu_top25_r_train_list.txt"
    # img_list_eu_test = "../Segmentation_Models/ICNet/list/eu_top25_r_test_list.txt"
    # img_list_eu_val = "../Segmentation_Models/ICNet/list/eu_top25_r_val_list.txt"
    # fd = evaluate_feature_distrib_from_GT(datadir=eu_set, img_list=None)
    # fd_train = evaluate_feature_distrib_from_GT(datadir=eu_set, img_list=img_list_eu_train)
    # fd_test = evaluate_feature_distrib_from_GT(datadir=eu_set, img_list=img_list_eu_test)
    # fd_val = evaluate_feature_distrib_from_GT(datadir=eu_set, img_list=img_list_eu_val)
    # print("eu_top25: complete")
    # print(fd)
    # print("eu_top25: train")
    # print(fd_train)
    # print("eu_top25: test")
    # print(fd_test)
    # print("eu_top25: val")
    # print(fd_val)
    # img_list_tiny_train = "../Segmentation_Models/ICNet/list/world_tiny2k_r_train_list.txt"
    # img_list_tiny_test = "../Segmentation_Models/ICNet/list/world_tiny2k_r_test_list.txt"
    # img_list_tiny_val = "../Segmentation_Models/ICNet/list/world_tiny2k_r_val_list.txt"
    # fd = evaluate_feature_distrib_from_GT(datadir=tiny_set, img_list=None)
    # fd_train = evaluate_feature_distrib_from_GT(datadir=tiny_set, img_list=img_list_tiny_train)
    # fd_test = evaluate_feature_distrib_from_GT(datadir=tiny_set, img_list=img_list_tiny_test)
    # fd_val = evaluate_feature_distrib_from_GT(datadir=tiny_set, img_list=img_list_tiny_val)
    # print("world_tiny2k: complete")
    # print(fd)
    # print("world_tiny2k: train")
    # print(fd_train)
    # print("world_tiny2k: test")
    # print(fd_test)
    # print("world_tiny2k: val")
    # print(fd_val)

    # SPLIT DATASETS TO TRAIN, TEST AND VAL
    # config = DefaultConfig()
    # config.custom_dir = tiny_set
    # config.custom_dir_gt = tiny_set+"/gt_data"
    # split = split_dataset(config, train=0.8, test=0.1, val=0.1, to_folder='.',ds_name="world_tiny2k_r")
    # config.custom_dir = de_set
    # config.custom_dir_gt = de_set+"/gt_data"
    # split = split_dataset(config, train=0.8, test=0.1, val=0.1, to_folder='.',ds_name="de_top14_r")
    # config.custom_dir = eu_set
    # config.custom_dir_gt = eu_set+"/gt_data"
    # split = split_dataset(config, train=0.8, test=0.1, val=0.1, to_folder='.',ds_name="eu_top25_exde_r")

    config = DefaultConfig()
    config.custom_dir = 'G:/Datasets/vaihingen'
    config.custom_dir_gt = 'G:/Datasets/vaihingen/gt_data'
    # config.custom_dir = tiny_set
    # config.custom_dir_gt = tiny_set+"/gt_data"
    split = split_dataset(config, train=0.9, test=0.05, val=0.05, to_folder='.', ds_name="vaihingen")

    # save_dataset_to_files(config)
    # crop_offset_from_dataset(config)
    # config.load_from_db = False

    # Eval Feature distrib of EU dataset
    # distrib = evaluate_feature_distrib_all(path="C:/Uni/Masterstudium/ma-werthmann/code/ANN_DATA/eu_top25_exde")
    # print(distrib)

    #config = DefaultConfig()
    #config.custom_dir=config.datadir+ '/duisburg_data_cropped'
    #config.num_images = 500

    # LOOP OVER ALL SAMPLES & PLOT
    config = DefaultConfig()
    config.num_images = 50
    config.custom_dir = '../ANN_DATA/eu_top25_exde_cropped'
    config.custom_dir_gt = '../ANN_DATA/eu_top25_exde_cropped/gt_data'
    config.load_from_db = True
    for i in range(0, config.num_images):
        image, train_mask, region = get_sample(config, get_region=True)
        # evaluate_feature_distrib(train_mask)  # test manual feature distrib eval
        fig, ax = plt.subplots(1, 3)
        # google image

        ax[0].imshow(image)
        ax[0].set_title('google satellite image: ' + region)

        # postgis ground-truth image
        ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
        ax[1].set_title('postgis-created ground-truth')

        # alpha overlay image
        alpha = create_alpha_layer(image, train_mask)
        ax[2].imshow(image, alpha=0.75)
        ax[2].imshow(alpha, alpha=0.25)
        ax[2].set_title('alpha blend')

        # show max zoomed plot (windwows TkAgg)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
        # maybe_download_images(config=DefaultConfig())

        # !Eval all feature distributions for all regions!
        # TAKES SOME TIME!
        # for pre in ["ber", "poz", "las", "mal", "unt"]:
        #     distrib = evaluate_feature_distrib_all(filter_prefix=pre)
        #     print(pre)
        #     print(distrib)
