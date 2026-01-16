CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.AMENITY_SIGNAL_TTEST AS
WITH assessor_ny AS (
    SELECT
        propertyid,
        situsstate,
        SALEPRICEAMOUNT as SALESAMT,
        TO_GEOGRAPHY(ST_MAKEPOINT(situslongitude, situslatitude)) AS house_point
    FROM ROC_PUBLIC_RECORD_DATA.DATATREE.ASSESSOR
    WHERE situsstate IN ('ny', 'oh', 'nc')
      AND SALESAMT IS NOT NULL
      AND SALESAMT > 0
),
amenities AS (
    SELECT
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
        ST_DISTANCE(hc.house_point, p.poi_point) AS dist_m
    FROM house_cells hc
    JOIN pois_h3 p ON p.h3 = hc.h3
),
nearest_amenities AS (
    SELECT
        propertyid,
        amenity,
        MIN(dist_m) AS nearest_dist_m
    FROM candidates
    GROUP BY 1, 2
),
properties_classified AS (
    SELECT
        a.SALESAMT as salesamt,
        na.amenity,
        na.nearest_dist_m,
        CASE
            WHEN na.nearest_dist_m <= 2000 THEN 'NEAR'
            WHEN na.nearest_dist_m > 5000 THEN 'FAR'
            ELSE 'MEDIUM'
        END AS distance_group
    FROM assessor_ny a
    INNER JOIN nearest_amenities na ON a.propertyid = na.propertyid
    WHERE na.amenity IS NOT NULL
),
amenity_stats AS (
    SELECT
        amenity,
        AVG(CASE WHEN distance_group = 'NEAR' THEN salesamt END) AS avg_price_near,
        STDDEV(CASE WHEN distance_group = 'NEAR' THEN salesamt END) AS std_price_near,
        COUNT(CASE WHEN distance_group = 'NEAR' THEN 1 END) AS n_near,
        AVG(CASE WHEN distance_group = 'FAR' THEN salesamt END) AS avg_price_far,
        STDDEV(CASE WHEN distance_group = 'FAR' THEN salesamt END) AS std_price_far,
        COUNT(CASE WHEN distance_group = 'FAR' THEN 1 END) AS n_far,
        AVG(CASE WHEN distance_group = 'NEAR' THEN salesamt END) -
        AVG(CASE WHEN distance_group = 'FAR' THEN salesamt END) AS price_difference,
        (AVG(CASE WHEN distance_group = 'NEAR' THEN salesamt END) -
         AVG(CASE WHEN distance_group = 'FAR' THEN salesamt END)) /
        SQRT(POWER(STDDEV(CASE WHEN distance_group = 'NEAR' THEN salesamt END), 2) /
             COUNT(CASE WHEN distance_group = 'NEAR' THEN 1 END) +
             POWER(STDDEV(CASE WHEN distance_group = 'FAR' THEN salesamt END), 2) /
             COUNT(CASE WHEN distance_group = 'FAR' THEN 1 END)) AS t_statistic,
        (AVG(CASE WHEN distance_group = 'NEAR' THEN salesamt END) -
         AVG(CASE WHEN distance_group = 'FAR' THEN salesamt END)) /
        SQRT((POWER(STDDEV(CASE WHEN distance_group = 'NEAR' THEN salesamt END), 2) *
              (COUNT(CASE WHEN distance_group = 'NEAR' THEN 1 END) - 1) +
              POWER(STDDEV(CASE WHEN distance_group = 'FAR' THEN salesamt END), 2) *
              (COUNT(CASE WHEN distance_group = 'FAR' THEN 1 END) - 1)) /
             (COUNT(CASE WHEN distance_group = 'NEAR' THEN 1 END) +
              COUNT(CASE WHEN distance_group = 'FAR' THEN 1 END) - 2)) AS cohens_d
    FROM properties_classified
    WHERE distance_group IN ('NEAR', 'FAR')
    GROUP BY amenity
    HAVING COUNT(CASE WHEN distance_group = 'NEAR' THEN 1 END) >= 100
       AND COUNT(CASE WHEN distance_group = 'FAR' THEN 1 END) >= 100
)
SELECT
    amenity,
    n_near,
    n_far,
    ROUND(avg_price_near, 0) AS avg_price_near,
    ROUND(avg_price_far, 0) AS avg_price_far,
    ROUND(price_difference, 0) AS price_difference,
    ROUND(100.0 * price_difference / NULLIF(avg_price_far, 0), 2) AS pct_difference,
    ROUND(cohens_d, 3) AS cohens_d,
    ROUND(t_statistic, 2) AS t_statistic,
    CASE
        WHEN ABS(t_statistic) >= 3.0 THEN 'VERY STRONG'
        WHEN ABS(t_statistic) >= 2.58 THEN 'STRONG'
        WHEN ABS(t_statistic) >= 1.96 THEN 'MODERATE'
        ELSE 'WEAK'
    END AS signal_strength
FROM amenity_stats
ORDER BY ABS(t_statistic) DESC;
