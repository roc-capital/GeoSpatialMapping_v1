WITH assessor_ny AS (
    SELECT
        propertyid,
        TO_GEOGRAPHY(ST_MAKEPOINT(situslongitude, situslatitude)) AS house_point
    FROM ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR
    WHERE situsstate = 'ny'
    LIMIT 100000
),
amenities AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY LAT, LON, AMENITY) AS poi_id,
        AMENITY,
        TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
    WHERE LAT IS NOT NULL
      AND LON IS NOT NULL
      AND AMENITY IS NOT NULL
),
houses_h3 AS (
    SELECT
        propertyid,
        house_point,
        H3_POINT_TO_CELL(house_point, 9) AS h3
    FROM assessor_ny
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
        h.propertyid,
        h.house_point,
        c.value::NUMBER AS h3
    FROM houses_h3 h,
    LATERAL FLATTEN(INPUT => H3_GRID_DISK(h.h3, 10)) c
),
candidates AS (
    SELECT
        hc.propertyid,
        p.amenity,
        p.poi_id,
        ST_DISTANCE(hc.house_point, p.poi_point) AS dist_m
    FROM house_cells hc
    JOIN pois_h3 p ON p.h3 = hc.h3
),
nearest_amenities AS (
    SELECT
        propertyid,
        amenity,
        MIN(dist_m) AS nearest_dist_m,
        MIN_BY(poi_id, dist_m) AS nearest_poi_id
    FROM candidates
    GROUP BY 1, 2
),
address_mapping AS (
    SELECT
        f.value::STRING AS MLS_ID,
        a.value::STRING AS ASSESSOR_ID,
        a.value::STRING AS PROPERTY_ID
    FROM ROC_ANALYTICS.ANALYTICS.ADDRESSES_MASTER,
    LATERAL FLATTEN(input => MLS_IDS) f,
    LATERAL FLATTEN(input => ASSESSOR_IDS) a
    GROUP BY MLS_ID, ASSESSOR_ID, PROPERTY_ID
),
mls_data AS (
    SELECT DISTINCT
        CC_LIST_ID,
        CC_PROPERTY_ID
    FROM SCRATCH.BASIS.STG_MLS
),
mls_to_assessor AS (
    SELECT
        am.MLS_ID,
        am.ASSESSOR_ID,
        am.PROPERTY_ID,
        m.CC_LIST_ID,
        m.CC_PROPERTY_ID
    FROM address_mapping am
    LEFT JOIN mls_data m
        ON am.MLS_ID = m.CC_PROPERTY_ID
    QUALIFY ROW_NUMBER() OVER (PARTITION BY m.CC_LIST_ID ORDER BY am.MLS_ID) = 1
)
-- Final join: MLS data with nearest amenities
SELECT
    mta.CC_LIST_ID,
    mta.CC_PROPERTY_ID,
    mta.MLS_ID,
    mta.ASSESSOR_ID,
    mta.PROPERTY_ID,
    na.amenity,
    na.nearest_dist_m,
    na.nearest_poi_id
FROM mls_to_assessor mta
LEFT JOIN nearest_amenities na
    ON mta.ASSESSOR_ID = na.propertyid
ORDER BY mta.CC_LIST_ID, na.amenity;