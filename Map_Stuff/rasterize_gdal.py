import ogr, osr
import gdal

###
###
###
# Draw Polygones with gdal: thx to https://gis.stackexchange.com/a/200634


# Setup working spatial reference
sr_wkt = 'LOCAL_CS["arbitrary"]'
sr = osr.SpatialReference()
sr.SetWellKnownGeogCS('WGS84')

# Create a memory raster to rasterize into.

target_ds = gdal.GetDriverByName('MEM').Create('', 200, 200, 3,
                                               gdal.GDT_Byte)
# https://gis.stackexchange.com/questions/165950/gdal-setgeotransform-does-not-work-as-expected
# target_ds.SetGeoTransform((1000, 1, 0, 1100, 0, -1))
target_ds.SetGeoTransform((9.948, 1, 0,  49.7600, 0, -1))
sr.ImportFromEPSG(3857)

# Create a memory layer to rasterize from.
rast_ogr_ds = \
          ogr.GetDriverByName('Memory').CreateDataSource('wrk')
rast_mem_lyr = rast_ogr_ds.CreateLayer('poly', srs=sr)

# Add a polygon with hole.
wkt_geom = 'POLYGON((1020 1076 ,1025 1085 ,1065 1090 ,1064 1078 ,1020 1076 ), (1023 1079 ,1061 1081 ,1062 1087 ,1028 1082 ,1023 1079 ))'
with open("file.json") as ft:
    data = ft.read()
mygeo = ogr.CreateGeometryFromJson(data)
feat1 = ogr.Feature(rast_mem_lyr.GetLayerDefn())
feat2 = ogr.Feature(rast_mem_lyr.GetLayerDefn())
# use wkt
feat1.SetGeometryDirectly(ogr.Geometry(wkt=wkt_geom))
# use geojson
feat2.SetGeometryDirectly(mygeo)

rast_mem_lyr.CreateFeature(feat1)
rast_mem_lyr.CreateFeature(feat2)
# Run the algorithm.

err = gdal.RasterizeLayer(target_ds, [3, 2, 1], rast_mem_lyr,
                           burn_values=[0, 0, 240])

gdal.GetDriverByName('PNG').CreateCopy('./rasterize_1.png', target_ds)

# export test
# dataset.GetLayer(0) POINT
# dataset.GetLayer(1) LINESTRING
# dataset.GetLayer(2) MULTILINESTRING
# dataset.GetLayer(3) MULTIPOLYGON
driver = ogr.GetDriverByName('OSM')
dataset = driver.Open(r'C:\Uni\Masterstudium\ma-werthmann\code\ANN_DATA\Google_Osm\wuerzburg_224\osm\wue_49.760077061614_9.94914493414561.osm')
layer = dataset.GetLayer(0)
spatialRef = layer.GetSpatialRef()
print(layer.GetLayerDefn())
feat = None
for feature in layer:

    geom = feature.GetGeometryRef().ExportToWkt()
    print(geom)
    feat = ogr.Feature(rast_mem_lyr.GetLayerDefn())
    # use wkt
    feat.SetGeometryDirectly(ogr.Geometry(wkt=geom))
    rast_mem_lyr.CreateFeature(feat)
    for f in rast_mem_lyr:
        print(f.GetGeomFieldCount())

err = gdal.RasterizeLayer(target_ds, [3, 2, 1], rast_mem_lyr,
                           burn_values=[0, 0, 240])

gdal.GetDriverByName('PNG').CreateCopy('./rasterize_1.png', target_ds)

