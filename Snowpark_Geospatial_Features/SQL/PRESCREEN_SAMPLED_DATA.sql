-- final dataframe before snowpark analysis
CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.GSF_SCREENING AS
WITH sample_properties AS (
  SELECT DISTINCT
    CC_PROPERTY_ID,
    ASSESSOR_ID,
    situsstate
  FROM SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_DATA
),
assessor_data AS (
  SELECT
    sp.CC_PROPERTY_ID,
    a.propertyid,
    a.CURRENTSALESPRICE,
    a.CURRENTSALECONTRACTDATE,
    TO_GEOGRAPHY(ST_MAKEPOINT(a.situslongitude, a.situslatitude)) AS house_point
  FROM sample_properties sp
  INNER JOIN ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR a
    ON sp.ASSESSOR_ID = a.propertyid
),
amenities AS (
  SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point, CONCAT(LAT, '_', LON, '_', AMENITY) AS poi_id
  FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
  WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL

  UNION ALL

  SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point, CONCAT(LAT, '_', LON, '_', AMENITY) AS poi_id
  FROM SCRATCH.OSM_DATA.OHIO_AMENITIES_V3
  WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL

  UNION ALL

  SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point, CONCAT(LAT, '_', LON, '_', AMENITY) AS poi_id
  FROM SCRATCH.OSM_DATA.NORTH_CAROLINA_AMENITIES_V3
  WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL
),
houses_h3 AS (
  SELECT
    CC_PROPERTY_ID,
    propertyid,
    CURRENTSALESPRICE,
    CURRENTSALECONTRACTDATE,
    house_point,
    H3_POINT_TO_CELL(house_point, 9) AS h3
  FROM assessor_data
),
pois_h3 AS (
  SELECT
    poi_id,
    amenity,
    poi_point,
    H3_POINT_TO_CELL(poi_point, 9) AS h3
  FROM amenities
),
house_cells AS (
  SELECT
    h.CC_PROPERTY_ID,
    h.propertyid,
    h.CURRENTSALESPRICE,
    h.CURRENTSALECONTRACTDATE,
    h.house_point,
    c.value::NUMBER AS h3
  FROM houses_h3 h,
  LATERAL FLATTEN(INPUT => H3_GRID_DISK(h.h3, 10)) c
),
candidates AS (
  SELECT
    hc.CC_PROPERTY_ID,
    hc.propertyid,
    hc.CURRENTSALESPRICE,
    hc.CURRENTSALECONTRACTDATE,
    p.amenity,
    p.poi_id,
    ST_DISTANCE(hc.house_point, p.poi_point) AS dist_m
  FROM house_cells hc
  JOIN pois_h3 p ON p.h3 = hc.h3
)
SELECT
  CC_PROPERTY_ID,
  propertyid,
  amenity,
  MIN(dist_m) AS nearest_dist_meters,
  MIN(dist_m) / 1609.34 AS nearest_dist_miles,
  MIN_BY(poi_id, dist_m) AS nearest_poi_id,
  MAX(CURRENTSALESPRICE) AS CURRENTSALESPRICE,
  MAX(CURRENTSALECONTRACTDATE) AS CURRENTSALECONTRACTDATE
FROM candidates
GROUP BY 1, 2, 3
ORDER BY CC_PROPERTY_ID, amenity;