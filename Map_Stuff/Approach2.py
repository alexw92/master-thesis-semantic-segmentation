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
    mapWidth = 640
    mapHeight = 640
    centerPoint = G_LatLng(centerLat, centerLon)
    centerPoint = G_LatLng(centerLat, centerLon)
    corners = getCorners(centerPoint, zoom, mapWidth, mapHeight)
    print(corners)

