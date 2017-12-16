# source: https://gist.github.com/robintw/9366322#file-osm_to_shapefile-py

import xml.etree.cElementTree as ET
from shapely.geometry import mapping, Polygon
import fiona		# install fiona https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona requires gdal!!
import os
import matplotlib.pyplot as plt
import shapefile    # pip install pyshp


# left, bottom, right, top
# bounds = b.attrib['minlat'], b.attrib['minlon'], b.attrib['maxlat'], b.attrib['maxlon']
def __convert_longLat_to_pixel(minX, minY, maxX, maxY, x, y, newXMax=300, newYMax=300):
    xdiff = maxX - minX
    ydiff = maxY - minY
    xratio =
    xscale = newXMax/xdiff
    yscale = newYMax/ydiff
    print(xscale * x)
    print(yscale * y)



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
bounds = b.attrib['minlat'], b.attrib['minlon'], b.attrib['maxlat'], b.attrib['maxlon']
xmin, ymin, xmax, ymax = bounds
print(bounds)
print(str(float(xmax) - float(xmin)), str(float(ymax) - float(ymin)))
__convert_longLat_to_pixel(float(xmin), float(ymin), float(xmax), float(ymax), 49.7490, 9.95)  # lat="49.7551035" lon="9.9579631"
# Put all nodes in a dictionary
for n in r.findall('node'):
    nodes[n.attrib['id']] = (float(n.attrib['lon']), float(n.attrib['lat']))


# Create a dictionary to hold the polygons we create
polygons = {}

# For each 'way' in the file
for way in r.findall("way"):

    coords = []

    # Look at all of the children of the <way> node
    for c in way.getchildren():
        if c.tag == "nd":
            # If it's a <nd> tag then it refers to a node, so get the lat-lon of that node
            # using the dictionary we created earlier and append it to the co-ordinate list
            ll = nodes[c.attrib['ref']]
           # print(ll)
            coords.append(ll)
        if c.tag == "tag":
            # If it's a <tag> tag (confusing, huh?) then it'll have some useful information in it for us
            # Get out the information we want as an attribute (in this case village name - which could be
            # in any of a number of fields, so we check)
            if c.attrib['v'] not in ("residential", 'village'):
                village_name = c.attrib['v']
    # Take the list of co-ordinates and convert to a Shapely polygon
    if len(coords) > 2:
        polygons[village_name] = Polygon(coords)


# Set up the schema for the shapefile
# In this case we have two columns: id and name
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int', 'name': 'str'},
}

# Remove the shapefile if it exists already
try:
    os.remove(output_filename)
except:
    pass

i = 0


# Write the shapefile
with fiona.open(output_filename, 'w', 'ESRI Shapefile', schema) as c:
    # For every polygon we stored earlier
    for name, p in polygons.items():
        i += 1
       #  print("Writing: %s" % name)
        # Write it to the shapefile
        c.write({
            'geometry': mapping(p),
            'properties': {'id': i, 'name': name},
            })

# plot shapefile source: https://gis.stackexchange.com/a/152331
listx = []
listy = []
sf = shapefile.Reader(output_filename)
plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y)
plt.show()
