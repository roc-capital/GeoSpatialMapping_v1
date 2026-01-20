CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_ALL_DATA_APPENDED AS
WITH base_data AS (
    SELECT *
    FROM SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_DATA
),
assessor_prices AS (
    SELECT
        propertyid,
        mostrecentsaledate,
        mostrecentsaleprice,
        TO_GEOGRAPHY(ST_MAKEPOINT(situslongitude, situslatitude)) AS house_point
    FROM ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR
    WHERE mostrecentsaleprice IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY propertyid ORDER BY mostrecentsaledate DESC) = 1
),
all_amenities AS (
    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL

    UNION ALL

    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.OHIO_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL

    UNION ALL

    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.NORTH_CAROLINA_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL
),
distinct_amenities AS (
    SELECT DISTINCT
        STATE,
        LAT,
        LON,
        AMENITY,
        TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM all_amenities
),
min_distances AS (
    SELECT
        bd.CC_PROPERTY_ID,
        MIN(ST_DISTANCE(ap.house_point, da.poi_point)) AS min_distance_meters
    FROM base_data bd
    INNER JOIN assessor_prices ap ON bd.ASSESSOR_ID = ap.propertyid
    INNER JOIN distinct_amenities da ON LOWER(bd.situsstate) = LOWER(da.STATE)
    GROUP BY bd.CC_PROPERTY_ID
)
SELECT
    bd.*,
    ap.mostrecentsaledate,
    ap.mostrecentsaleprice,
    md.min_distance_meters
FROM base_data bd
LEFT JOIN assessor_prices ap ON bd.ASSESSOR_ID = ap.propertyid
LEFT JOIN min_distances md ON bd.CC_PROPERTY_ID = md.CC_PROPERTY_ID
ORDER BY bd.CC_PROPERTY_ID, bd.amenity;


-------

CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_DATA_ENHANCED AS
WITH base_data AS (
    SELECT *
    FROM SCRATCH.OSM_DATA.AMENITY_COVERAGE_SAMPLE_DATA
),
all_amenities AS (
    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.NEW_YORK_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL

    UNION ALL

    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.OHIO_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL

    UNION ALL

    SELECT STATE, LAT, LON, AMENITY
    FROM SCRATCH.OSM_DATA.NORTH_CAROLINA_AMENITIES_V3
    WHERE LAT IS NOT NULL AND LON IS NOT NULL
),
distinct_amenities AS (
    SELECT DISTINCT
        STATE,
        LAT,
        LON,
        AMENITY,
        TO_GEOGRAPHY(ST_MAKEPOINT(LON, LAT)) AS poi_point
    FROM all_amenities
),
properties_with_prices AS (
    SELECT
        bd.CC_PROPERTY_ID,
        bd.ASSESSOR_ID,
        bd.SITUSSTATE,
        bd.AMENITY,
        a.MOSTRECENTSALEDATE,
        a.MOSTRECENTSALEPRICE,
        TO_GEOGRAPHY(ST_MAKEPOINT(a.SITUSLONGITUDE, a.SITUSLATITUDE)) AS house_point
    FROM base_data bd
    LEFT JOIN ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR a
        ON bd.ASSESSOR_ID = a.PROPERTYID
),
min_distances AS (
    SELECT
        pwp.CC_PROPERTY_ID,
        MIN(ST_DISTANCE(pwp.house_point, da.poi_point)) AS MIN_DISTANCE_METERS
    FROM properties_with_prices pwp
    INNER JOIN distinct_amenities da
        ON LOWER(pwp.SITUSSTATE) = LOWER(da.STATE)
    WHERE pwp.house_point IS NOT NULL
    GROUP BY pwp.CC_PROPERTY_ID
)
SELECT
    pwp.CC_PROPERTY_ID,
    pwp.ASSESSOR_ID,
    pwp.SITUSSTATE,
    pwp.AMENITY,
    pwp.MOSTRECENTSALEDATE,
    pwp.MOSTRECENTSALEPRICE,
    md.MIN_DISTANCE_METERS
FROM properties_with_prices pwp
LEFT JOIN min_distances md
    ON pwp.CC_PROPERTY_ID = md.CC_PROPERTY_ID
ORDER BY pwp.CC_PROPERTY_ID, pwp.AMENITY;
