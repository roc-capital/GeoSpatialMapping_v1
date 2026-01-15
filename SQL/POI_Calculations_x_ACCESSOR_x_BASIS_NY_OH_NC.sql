CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.NY_AMENITIES_H3 AS
SELECT
    ROW_NUMBER() OVER (ORDER BY LAT, LON, AMENITY) AS poi_id,
    AMENITY,
    LAT,
    LON,
    TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point,
    H3_POINT_TO_CELL(TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)), 9) AS h3
FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL;

-- 2. Create materialized assessor H3 table
CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.NY_ASSESSOR_H3 AS
SELECT
    a.propertyid,
    TO_GEOGRAPHY(ST_MAKEPOINT(a.situslongitude, a.situslatitude)) AS house_point,
    H3_POINT_TO_CELL(TO_GEOGRAPHY(ST_MAKEPOINT(a.situslongitude, a.situslatitude)), 9) AS h3
FROM ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR a
WHERE a.situsstate = 'ny'
  AND a.situslongitude IS NOT NULL
  AND a.situslatitude IS NOT NULL;

-- 3. Create materialized MLS to Assessor mapping
CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.MLS_ASSESSOR_MAPPING AS
SELECT
    f.value::STRING AS MLS_ID,
    a.value::STRING AS ASSESSOR_ID
FROM ROC_ANALYTICS.ANALYTICS.ADDRESSES_MASTER,
LATERAL FLATTEN(input => MLS_IDS) f,
LATERAL FLATTEN(input => ASSESSOR_IDS) a
WHERE f.value IS NOT NULL AND a.value IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY f.value::STRING ORDER BY a.value::STRING) = 1;

-- 4. Now the optimized query is much simpler:
WITH house_cells AS (
    SELECT
        h.propertyid,
        h.house_point,
        c.value::NUMBER AS h3_cell
    FROM SCRATCH.OSM_DATA.NY_ASSESSOR_H3 h,
    LATERAL FLATTEN(INPUT => H3_GRID_DISK(h.h3, 10)) c
),
candidates AS (
    SELECT
        hc.propertyid,
        p.amenity,
        p.poi_id,
        ST_DISTANCE(hc.house_point, p.poi_point) AS dist_m
    FROM house_cells hc
    INNER JOIN SCRATCH.OSM_DATA.NY_AMENITIES_H3 p
        ON p.h3 = hc.h3_cell
),
nearest_amenities AS (
    SELECT
        propertyid,
        amenity,
        MIN(dist_m) AS nearest_dist_m,
        MIN_BY(poi_id, dist_m) AS nearest_poi_id
    FROM candidates
    GROUP BY propertyid, amenity
)
SELECT
    b.*,
    map.ASSESSOR_ID,
    na.amenity,
    na.nearest_dist_m,
    na.nearest_poi_id
FROM SCRATCH.BASIS.STG_MLS b
INNER JOIN SCRATCH.OSM_DATA.MLS_ASSESSOR_MAPPING map
    ON b.CC_PROPERTY_ID = map.MLS_ID
INNER JOIN nearest_amenities na
    ON map.ASSESSOR_ID = na.propertyid
ORDER BY b.CC_LIST_ID, na.amenity;