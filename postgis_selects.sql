-- Geom (ist ein hexa wbt format) in wkt umwandeln
-- http://postgis.net/docs/manual-1.5/ch04.html#OpenGISWKBWKT
-- https://gis.stackexchange.com/questions/146266/understanding-the-format-wkb-from-wkt-and-how-to-convert-the-first-into-the-lat
-- BERLIN GEO-DATA Boundingbox:
-- POLYGON((13.183786862415506 52.60627116138647, 13.183786862415506 52.39727845787755, 13.62735986046238 52.39727845787755, 13.62735986046238 52.60627116138647, 13.183786862415506 52.60627116138647))

-- MALAGA SMALL GEO-DATA Boudingbox:
-- POLYGON((-4.524576310773522 36.7584306448112, -4.524576310773522 36.681650962207854, -4.2324086471993025 36.681650962207854, -4.2324086471993025 36.7584306448112, -4.524576310773522 36.7584306448112))

-- MISSING: DIREKT INTERSECTION MIT BOUNDINGBOX-POLYGON ERMITTELN UND ZURÜCKGEBEN, Polygon für Unterfrankendaten aus geojson
-- POLYGON((9.086199142965881 50.07247606571221, 9.086199142965881 49.66887056979786, 10.61879191640338 49.66887056979786, 10.61879191640338 50.07247606571221, 9.086199142965881 50.07247606571221))

-- Buildings
SELECT building,ST_AsText(geo)
FROM ( SELECT building, geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys WHERE polys.building IS NOT NULL  LIMIT 10) AS f GROUP BY building, f.geo;

-- Roads

-- SELECT highway, width, ST_AsText(geo) 
-- FROM ( SELECT highway, width, geom AS geo FROM unterfranken_planet_osm_line_lines AS lines WHERE lines.highway IS NOT NULL  LIMIT 100) AS f GROUP BY highway, width, f.geo ;


-- Water

SELECT water,ST_AsText(geo) 
FROM ( SELECT water, geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys WHERE polys.water IS NOT NULL LIMIT 10) AS f GROUP BY water, f.geo ;

SELECT waterway,ST_AsText(geo) 
FROM (SELECT waterway, geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys
WHERE polys.waterway IS NOT NULL LIMIT 10) AS f GROUP BY waterway, f.geo ;

SELECT "natural", ST_AsText(geo) 
FROM (SELECT "natural", geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys
WHERE polys."natural" = 'water' LIMIT 10) AS f GROUP BY "natural", f.geo ;

-- Wood

SELECT "natural", ST_AsText(geo) 
FROM (SELECT "natural", geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys
WHERE polys."natural" = 'wood' LIMIT 100) AS f GROUP BY "natural", f.geo ;

SELECT landuse, ST_AsText(geo) 
FROM (SELECT landuse, geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys
WHERE polys.landuse = 'forest' LIMIT 100) AS f GROUP BY landuse, f.geo ;

SELECT landcover, ST_AsText(geo) 
FROM (SELECT landcover, geom AS geo FROM unterfranken_planet_osm_polygon_polygons AS polys
WHERE polys.landcover = 'trees' LIMIT 100) AS f GROUP BY landcover, f.geo ;

------------------------
-- Sample Request for Getting Area percentage for wood feature
SELECT SUM(ST_Area(xyz.geo))/ST_Area(ST_GeomFromText('POLYGON((9.086199142965881 50.07247606571221, 9.086199142965881 50.00887056979786, 9.14879191640338 50.00887056979786, 9.14879191640338 50.07247606571221, 9.086199142965881 50.07247606571221))')) AS polybuilding
FROM (
SELECT ST_Intersection('POLYGON((9.086199142965881 50.07247606571221, 9.086199142965881 50.00887056979786, 9.14879191640338 50.00887056979786, 9.14879191640338 50.07247606571221, 9.086199142965881 50.07247606571221))',geom) AS geo
FROM unterfranken_planet_osm_polygon_polygons
WHERE ( 'natural'='wood' OR landuse='forest')
AND ST_Intersects('POLYGON((9.086199142965881 50.07247606571221, 9.086199142965881 50.00887056979786, 9.14879191640338 50.00887056979786, 9.14879191640338 50.07247606571221, 9.086199142965881 50.07247606571221))',geom)) AS xyz
;
