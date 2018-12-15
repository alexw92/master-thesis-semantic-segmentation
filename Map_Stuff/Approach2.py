# This code calculates the bbox from a google static map image
# when centre long&lat, zoom-factor and map width and height in pixels
# Thanks to jmague : https://stackoverflow.com/a/21116468/8862202
from __future__ import division
import math
MERCATOR_RANGE = 256

def  bound(value, opt_min, opt_max):
  if (opt_min != None): 
    value = max(value, opt_min)
  if (opt_max != None): 
    value = min(value, opt_max)
  return value


def  degreesToRadians(deg) :
  return deg * (math.pi / 180)


def  radiansToDegrees(rad) :
  return rad / (math.pi / 180)


class G_Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class G_LatLng:
    def __init__(self,lt, ln):
        self.lat = lt
        self.lng = ln


class MercatorProjection :

    def __init__(self) :
      self.pixelOrigin_ =  G_Point( MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
      self.pixelsPerLonDegree_ = MERCATOR_RANGE / 360
      self.pixelsPerLonRadian_ = MERCATOR_RANGE / (2 * math.pi)

    def fromLatLngToPoint(self, latLng, opt_point=None) :
      point = opt_point if opt_point is not None else G_Point(0,0)
      origin = self.pixelOrigin_
      point.x = origin.x + latLng.lng * self.pixelsPerLonDegree_
      # NOTE(appleton): Truncating to 0.9999 effectively limits latitude to
      # 89.189.  This is about a third of a tile past the edge of the world tile.
      siny = bound(math.sin(degreesToRadians(latLng.lat)), -0.9999, 0.9999)
      point.y = origin.y + 0.5 * math.log((1 + siny) / (1 - siny)) * -     self.pixelsPerLonRadian_
      return point

    def fromPointToLatLng(self,point) :
        origin = self.pixelOrigin_
        lng = (point.x - origin.x) / self.pixelsPerLonDegree_
        latRadians = (point.y - origin.y) / -self.pixelsPerLonRadian_
        lat = radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
        return G_LatLng(lat, lng)

# pixelCoordinate = worldCoordinate * pow(2,zoomLevel)


def getCorners(center, zoom, mapWidth, mapHeight):
    scale = 2**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(center)
    print(centerPx.x)
    print(centerPx.y)
    SWPoint = G_Point(centerPx.x-(mapWidth/2)/scale, centerPx.y+(mapHeight/2)/scale)
    SWLatLon = proj.fromPointToLatLng(SWPoint)
    NEPoint = G_Point(centerPx.x+(mapWidth/2)/scale, centerPx.y-(mapHeight/2)/scale)
    NELatLon = proj.fromPointToLatLng(NEPoint)
    return {
        'N': NELatLon.lat,
        'E': NELatLon.lng,
        'S': SWLatLon.lat,
        'W': SWLatLon.lng,
    }


# from https://stackoverflow.com/a/47579340/8862202
# neglects map scale on y axis
# Seems to work as well
def get_static_map_bounds(lat, lng, zoom, sx, sy):
    # lat, lng - center
    # sx, sy - map size in pixels

    # 256 pixels - initial map size for zoom factor 0
    sz = 256 * 2 ** zoom

    #resolution in degrees per pixel
    res_lat = math.cos(lat * math.pi / 180.) * 360. / sz
    res_lng = 360./sz

    d_lat = res_lat * sy / 2
    d_lng = res_lng * sx / 2

    return {'W' :lng-d_lng, 'E' :lng+d_lng, 'S' : lat-d_lat, 'N' :lat+d_lat}


# thx 2 https://stackoverflow.com/a/19412565/8862202
def getMeterDistance(lon1, lat1, lon2, lat2):
    from math import sin, cos, sqrt, atan2, radians
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance*1000

# Usage
# centerLat = 49.141404
# centerLon = -121.960988
# würzburg heuchelhof
# gmaps https://maps.googleapis.com/maps/api/staticmap?center=49.7513,9.9609&size=800x800&zoom=16&maptype=hybrid
# how to download osm data with given bbox, order: left, bottom, right, top (min long, min lat, max long, max lat):
# http://api.openstreetmap.org/api/0.6/map?bbox=11.54,48.14,11.543,48.145
if __name__ == '__main__':
    centerLat = 49.7513
    centerLon = 9.9609
    zoom = 16
    mapWidth = 400
    mapHeight = 800
    centerPoint = G_LatLng(centerLat, centerLon)
    centerPoint = G_LatLng(centerLat, centerLon)
    corners = getCorners(centerPoint, zoom, mapWidth, mapHeight)

    print(corners)

    # second approach
    corners2 = get_static_map_bounds(centerLat, centerLon, zoom, mapWidth, mapHeight)
    print(corners2)
