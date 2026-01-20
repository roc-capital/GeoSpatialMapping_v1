CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_DATA AS
WITH address_mapping AS (
    SELECT
        f.value::STRING AS MLS_ID,
        TRY_CAST(a.value::STRING AS NUMBER) AS ASSESSOR_ID
    FROM ROC_ANALYTICS.ANALYTICS.ADDRESSES_MASTER,
    LATERAL FLATTEN(input => MLS_IDS) f,
    LATERAL FLATTEN(input => ASSESSOR_IDS) a
    WHERE a.value IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY f.value::STRING ORDER BY a.value::STRING) = 1
),
basis_with_assessor AS (
    SELECT
        b.CC_PROPERTY_ID,
        map.ASSESSOR_ID,
        a.situsstate,
        TO_GEOGRAPHY(ST_MAKEPOINT(a.situslongitude, a.situslatitude)) AS house_point
    FROM SCRATCH.BASIS.STG_MLS b
    INNER JOIN address_mapping map
        ON b.CC_PROPERTY_ID = map.MLS_ID
    INNER JOIN ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR a
        ON map.ASSESSOR_ID = a.propertyid
    WHERE a.situsstate IN ('ny', 'oh', 'nc')
),
sampled_basis_properties AS (
    SELECT
        CC_PROPERTY_ID,
        ASSESSOR_ID,
        situsstate,
        house_point
    FROM basis_with_assessor
    QUALIFY ROW_NUMBER() OVER (PARTITION BY situsstate ORDER BY RANDOM()) <= 9000
),
-- rest of your query stays the same
amenities AS (
    SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL

    UNION ALL

    SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM SCRATCH.OSM_DATA.OHIO_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL

    UNION ALL

    SELECT AMENITY, TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM SCRATCH.OSM_DATA.NORTH_CAROLINA_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL AND AMENITY IS NOT NULL
),
houses_h3 AS (
    SELECT
        CC_PROPERTY_ID,
        ASSESSOR_ID,
        situsstate,
        house_point,
        H3_POINT_TO_CELL(house_point, 9) AS h3
    FROM sampled_basis_properties
),
pois_h3 AS (
    SELECT
        amenity,
        poi_point,
        H3_POINT_TO_CELL(poi_point, 9) AS h3
    FROM amenities
),
house_cells AS (
    SELECT
        h.CC_PROPERTY_ID,
        h.ASSESSOR_ID,
        h.situsstate,
        c.value::NUMBER AS h3
    FROM houses_h3 h,
    LATERAL FLATTEN(INPUT => H3_GRID_DISK(h.h3, 10)) c
),
property_has_amenity AS (
    SELECT DISTINCT
        hc.CC_PROPERTY_ID,
        hc.ASSESSOR_ID,
        hc.situsstate,
        p.amenity
    FROM house_cells hc
    JOIN pois_h3 p ON p.h3 = hc.h3
)
SELECT
    CC_PROPERTY_ID,
    ASSESSOR_ID,
    situsstate,
    amenity
FROM property_has_amenity
ORDER BY CC_PROPERTY_ID, amenity;