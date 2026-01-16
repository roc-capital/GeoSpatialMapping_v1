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
amenities_pivoted AS (
    SELECT
        TRY_CAST(propertyid AS NUMBER) AS propertyid,
        MAX(CASE WHEN amenity = 'Retail' THEN nearest_dist_m END) AS retail_dist_m,
        MAX(CASE WHEN amenity = 'Retail' THEN nearest_poi_id END) AS retail_poi_id,
        MAX(CASE WHEN amenity = 'airport' THEN nearest_dist_m END) AS airport_dist_m,
        MAX(CASE WHEN amenity = 'airport' THEN nearest_poi_id END) AS airport_poi_id,
        MAX(CASE WHEN amenity = 'atm' THEN nearest_dist_m END) AS atm_dist_m,
        MAX(CASE WHEN amenity = 'atm' THEN nearest_poi_id END) AS atm_poi_id,
        MAX(CASE WHEN amenity = 'bar' THEN nearest_dist_m END) AS bar_dist_m,
        MAX(CASE WHEN amenity = 'bar' THEN nearest_poi_id END) AS bar_poi_id,
        MAX(CASE WHEN amenity = 'beach' THEN nearest_dist_m END) AS beach_dist_m,
        MAX(CASE WHEN amenity = 'beach' THEN nearest_poi_id END) AS beach_poi_id,
        MAX(CASE WHEN amenity = 'bench' THEN nearest_dist_m END) AS bench_dist_m,
        MAX(CASE WHEN amenity = 'bench' THEN nearest_poi_id END) AS bench_poi_id,
        MAX(CASE WHEN amenity = 'bicycle_parking' THEN nearest_dist_m END) AS bicycle_parking_dist_m,
        MAX(CASE WHEN amenity = 'bicycle_parking' THEN nearest_poi_id END) AS bicycle_parking_poi_id,
        MAX(CASE WHEN amenity = 'bicycle_rental' THEN nearest_dist_m END) AS bicycle_rental_dist_m,
        MAX(CASE WHEN amenity = 'bicycle_rental' THEN nearest_poi_id END) AS bicycle_rental_poi_id,
        MAX(CASE WHEN amenity = 'bus_station' THEN nearest_dist_m END) AS bus_station_dist_m,
        MAX(CASE WHEN amenity = 'bus_station' THEN nearest_poi_id END) AS bus_station_poi_id,
        MAX(CASE WHEN amenity = 'cafe' THEN nearest_dist_m END) AS cafe_dist_m,
        MAX(CASE WHEN amenity = 'cafe' THEN nearest_poi_id END) AS cafe_poi_id,
        MAX(CASE WHEN amenity = 'casino' THEN nearest_dist_m END) AS casino_dist_m,
        MAX(CASE WHEN amenity = 'casino' THEN nearest_poi_id END) AS casino_poi_id,
        MAX(CASE WHEN amenity = 'charging_station' THEN nearest_dist_m END) AS charging_station_dist_m,
        MAX(CASE WHEN amenity = 'charging_station' THEN nearest_poi_id END) AS charging_station_poi_id,
        MAX(CASE WHEN amenity = 'clinic' THEN nearest_dist_m END) AS clinic_dist_m,
        MAX(CASE WHEN amenity = 'clinic' THEN nearest_poi_id END) AS clinic_poi_id,
        MAX(CASE WHEN amenity = 'clock' THEN nearest_dist_m END) AS clock_dist_m,
        MAX(CASE WHEN amenity = 'clock' THEN nearest_poi_id END) AS clock_poi_id,
        MAX(CASE WHEN amenity = 'college' THEN nearest_dist_m END) AS college_dist_m,
        MAX(CASE WHEN amenity = 'college' THEN nearest_poi_id END) AS college_poi_id,
        MAX(CASE WHEN amenity = 'community_centre' THEN nearest_dist_m END) AS community_centre_dist_m,
        MAX(CASE WHEN amenity = 'community_centre' THEN nearest_poi_id END) AS community_centre_poi_id,
        MAX(CASE WHEN amenity = 'convenience_store' THEN nearest_dist_m END) AS convenience_store_dist_m,
        MAX(CASE WHEN amenity = 'convenience_store' THEN nearest_poi_id END) AS convenience_store_poi_id,
        MAX(CASE WHEN amenity = 'coworking_space' THEN nearest_dist_m END) AS coworking_space_dist_m,
        MAX(CASE WHEN amenity = 'coworking_space' THEN nearest_poi_id END) AS coworking_space_poi_id,
        MAX(CASE WHEN amenity = 'dentist' THEN nearest_dist_m END) AS dentist_dist_m,
        MAX(CASE WHEN amenity = 'dentist' THEN nearest_poi_id END) AS dentist_poi_id,
        MAX(CASE WHEN amenity = 'dojo' THEN nearest_dist_m END) AS dojo_dist_m,
        MAX(CASE WHEN amenity = 'dojo' THEN nearest_poi_id END) AS dojo_poi_id,
        MAX(CASE WHEN amenity = 'drinking_water' THEN nearest_dist_m END) AS drinking_water_dist_m,
        MAX(CASE WHEN amenity = 'drinking_water' THEN nearest_poi_id END) AS drinking_water_poi_id,
        MAX(CASE WHEN amenity = 'fast_food' THEN nearest_dist_m END) AS fast_food_dist_m,
        MAX(CASE WHEN amenity = 'fast_food' THEN nearest_poi_id END) AS fast_food_poi_id,
        MAX(CASE WHEN amenity = 'ferry_terminal' THEN nearest_dist_m END) AS ferry_terminal_dist_m,
        MAX(CASE WHEN amenity = 'ferry_terminal' THEN nearest_poi_id END) AS ferry_terminal_poi_id,
        MAX(CASE WHEN amenity = 'fire_station' THEN nearest_dist_m END) AS fire_station_dist_m,
        MAX(CASE WHEN amenity = 'fire_station' THEN nearest_poi_id END) AS fire_station_poi_id,
        MAX(CASE WHEN amenity = 'fitness_centre' THEN nearest_dist_m END) AS fitness_centre_dist_m,
        MAX(CASE WHEN amenity = 'fitness_centre' THEN nearest_poi_id END) AS fitness_centre_poi_id,
        MAX(CASE WHEN amenity = 'fountain' THEN nearest_dist_m END) AS fountain_dist_m,
        MAX(CASE WHEN amenity = 'fountain' THEN nearest_poi_id END) AS fountain_poi_id,
        MAX(CASE WHEN amenity = 'fuel' THEN nearest_dist_m END) AS fuel_dist_m,
        MAX(CASE WHEN amenity = 'fuel' THEN nearest_poi_id END) AS fuel_poi_id,
        MAX(CASE WHEN amenity = 'give_box' THEN nearest_dist_m END) AS give_box_dist_m,
        MAX(CASE WHEN amenity = 'give_box' THEN nearest_poi_id END) AS give_box_poi_id,
        MAX(CASE WHEN amenity = 'grave_yard' THEN nearest_dist_m END) AS grave_yard_dist_m,
        MAX(CASE WHEN amenity = 'grave_yard' THEN nearest_poi_id END) AS grave_yard_poi_id,
        MAX(CASE WHEN amenity = 'hospital' THEN nearest_dist_m END) AS hospital_dist_m,
        MAX(CASE WHEN amenity = 'hospital' THEN nearest_poi_id END) AS hospital_poi_id,
        MAX(CASE WHEN amenity = 'ice_cream' THEN nearest_dist_m END) AS ice_cream_dist_m,
        MAX(CASE WHEN amenity = 'ice_cream' THEN nearest_poi_id END) AS ice_cream_poi_id,
        MAX(CASE WHEN amenity = 'industrial' THEN nearest_dist_m END) AS industrial_dist_m,
        MAX(CASE WHEN amenity = 'industrial' THEN nearest_poi_id END) AS industrial_poi_id,
        MAX(CASE WHEN amenity = 'kindergarten' THEN nearest_dist_m END) AS kindergarten_dist_m,
        MAX(CASE WHEN amenity = 'kindergarten' THEN nearest_poi_id END) AS kindergarten_poi_id,
        MAX(CASE WHEN amenity = 'kitchen' THEN nearest_dist_m END) AS kitchen_dist_m,
        MAX(CASE WHEN amenity = 'kitchen' THEN nearest_poi_id END) AS kitchen_poi_id,
        MAX(CASE WHEN amenity = 'lake' THEN nearest_dist_m END) AS lake_dist_m,
        MAX(CASE WHEN amenity = 'lake' THEN nearest_poi_id END) AS lake_poi_id,
        MAX(CASE WHEN amenity = 'letter_box' THEN nearest_dist_m END) AS letter_box_dist_m,
        MAX(CASE WHEN amenity = 'letter_box' THEN nearest_poi_id END) AS letter_box_poi_id,
        MAX(CASE WHEN amenity = 'library' THEN nearest_dist_m END) AS library_dist_m,
        MAX(CASE WHEN amenity = 'library' THEN nearest_poi_id END) AS library_poi_id,
        MAX(CASE WHEN amenity = 'library_dropoff' THEN nearest_dist_m END) AS library_dropoff_dist_m,
        MAX(CASE WHEN amenity = 'library_dropoff' THEN nearest_poi_id END) AS library_dropoff_poi_id,
        MAX(CASE WHEN amenity = 'loading_dock' THEN nearest_dist_m END) AS loading_dock_dist_m,
        MAX(CASE WHEN amenity = 'loading_dock' THEN nearest_poi_id END) AS loading_dock_poi_id,
        MAX(CASE WHEN amenity = 'money_transfer' THEN nearest_dist_m END) AS money_transfer_dist_m,
        MAX(CASE WHEN amenity = 'money_transfer' THEN nearest_poi_id END) AS money_transfer_poi_id,
        MAX(CASE WHEN amenity = 'music_school' THEN nearest_dist_m END) AS music_school_dist_m,
        MAX(CASE WHEN amenity = 'music_school' THEN nearest_poi_id END) AS music_school_poi_id,
        MAX(CASE WHEN amenity = 'nightclub' THEN nearest_dist_m END) AS nightclub_dist_m,
        MAX(CASE WHEN amenity = 'nightclub' THEN nearest_poi_id END) AS nightclub_poi_id,
        MAX(CASE WHEN amenity = 'parking' THEN nearest_dist_m END) AS parking_dist_m,
        MAX(CASE WHEN amenity = 'parking' THEN nearest_poi_id END) AS parking_poi_id,
        MAX(CASE WHEN amenity = 'parking_entrance' THEN nearest_dist_m END) AS parking_entrance_dist_m,
        MAX(CASE WHEN amenity = 'parking_entrance' THEN nearest_poi_id END) AS parking_entrance_poi_id,
        MAX(CASE WHEN amenity = 'parking_space' THEN nearest_dist_m END) AS parking_space_dist_m,
        MAX(CASE WHEN amenity = 'parking_space' THEN nearest_poi_id END) AS parking_space_poi_id,
        MAX(CASE WHEN amenity = 'peak' THEN nearest_dist_m END) AS peak_dist_m,
        MAX(CASE WHEN amenity = 'peak' THEN nearest_poi_id END) AS peak_poi_id,
        MAX(CASE WHEN amenity = 'pharmacy' THEN nearest_dist_m END) AS pharmacy_dist_m,
        MAX(CASE WHEN amenity = 'pharmacy' THEN nearest_poi_id END) AS pharmacy_poi_id,
        MAX(CASE WHEN amenity = 'place_of_worship' THEN nearest_dist_m END) AS place_of_worship_dist_m,
        MAX(CASE WHEN amenity = 'place_of_worship' THEN nearest_poi_id END) AS place_of_worship_poi_id,
        MAX(CASE WHEN amenity = 'post_box' THEN nearest_dist_m END) AS post_box_dist_m,
        MAX(CASE WHEN amenity = 'post_box' THEN nearest_poi_id END) AS post_box_poi_id,
        MAX(CASE WHEN amenity = 'pub' THEN nearest_dist_m END) AS pub_dist_m,
        MAX(CASE WHEN amenity = 'pub' THEN nearest_poi_id END) AS pub_poi_id,
        MAX(CASE WHEN amenity = 'railway_station' THEN nearest_dist_m END) AS railway_station_dist_m,
        MAX(CASE WHEN amenity = 'railway_station' THEN nearest_poi_id END) AS railway_station_poi_id,
        MAX(CASE WHEN amenity = 'restaurant' THEN nearest_dist_m END) AS restaurant_dist_m,
        MAX(CASE WHEN amenity = 'restaurant' THEN nearest_poi_id END) AS restaurant_poi_id,
        MAX(CASE WHEN amenity = 'school' THEN nearest_dist_m END) AS school_dist_m,
        MAX(CASE WHEN amenity = 'school' THEN nearest_poi_id END) AS school_poi_id,
        MAX(CASE WHEN amenity = 'shelter' THEN nearest_dist_m END) AS shelter_dist_m,
        MAX(CASE WHEN amenity = 'shelter' THEN nearest_poi_id END) AS shelter_poi_id,
        MAX(CASE WHEN amenity = 'sports_center' THEN nearest_dist_m END) AS sports_center_dist_m,
        MAX(CASE WHEN amenity = 'sports_center' THEN nearest_poi_id END) AS sports_center_poi_id,
        MAX(CASE WHEN amenity = 'swimming_pool' THEN nearest_dist_m END) AS swimming_pool_dist_m,
        MAX(CASE WHEN amenity = 'swimming_pool' THEN nearest_poi_id END) AS swimming_pool_poi_id,
        MAX(CASE WHEN amenity = 'theatre' THEN nearest_dist_m END) AS theatre_dist_m,
        MAX(CASE WHEN amenity = 'theatre' THEN nearest_poi_id END) AS theatre_poi_id,
        MAX(CASE WHEN amenity = 'toilets' THEN nearest_dist_m END) AS toilets_dist_m,
        MAX(CASE WHEN amenity = 'toilets' THEN nearest_poi_id END) AS toilets_poi_id,
        MAX(CASE WHEN amenity = 'university' THEN nearest_dist_m END) AS university_dist_m,
        MAX(CASE WHEN amenity = 'university' THEN nearest_poi_id END) AS university_poi_id,
        MAX(CASE WHEN amenity = 'vending_machine' THEN nearest_dist_m END) AS vending_machine_dist_m,
        MAX(CASE WHEN amenity = 'vending_machine' THEN nearest_poi_id END) AS vending_machine_poi_id,
        MAX(CASE WHEN amenity = 'veterinary' THEN nearest_dist_m END) AS veterinary_dist_m,
        MAX(CASE WHEN amenity = 'veterinary' THEN nearest_poi_id END) AS veterinary_poi_id,
        MAX(CASE WHEN amenity = 'waste_basket' THEN nearest_dist_m END) AS waste_basket_dist_m,
        MAX(CASE WHEN amenity = 'waste_basket' THEN nearest_poi_id END) AS waste_basket_poi_id,
        MAX(CASE WHEN amenity = 'waste_disposal' THEN nearest_dist_m END) AS waste_disposal_dist_m,
        MAX(CASE WHEN amenity = 'waste_disposal' THEN nearest_poi_id END) AS waste_disposal_poi_id,
        MAX(CASE WHEN amenity = 'wetland' THEN nearest_dist_m END) AS wetland_dist_m,
        MAX(CASE WHEN amenity = 'wetland' THEN nearest_poi_id END) AS wetland_poi_id
    FROM nearest_amenities
    GROUP BY TRY_CAST(propertyid AS NUMBER)
),
address_mapping AS (
    SELECT
        f.value::STRING AS MLS_ID,
        TRY_CAST(a.value::STRING AS NUMBER) AS ASSESSOR_ID
    FROM ROC_ANALYTICS.ANALYTICS.ADDRESSES_MASTER,
    LATERAL FLATTEN(input => MLS_IDS) f,
    LATERAL FLATTEN(input => ASSESSOR_IDS) a
    WHERE a.value IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY f.value::STRING ORDER BY a.value::STRING) = 1
)
-- Final join with proper NUMBER casting (134 amenity columns total)
SELECT
    b.*,
    map.ASSESSOR_ID,
    ap.*
FROM SCRATCH.BASIS.STG_MLS b
LEFT JOIN address_mapping map
    ON b.CC_PROPERTY_ID = map.MLS_ID
LEFT JOIN amenities_pivoted ap
    ON map.ASSESSOR_ID = ap.propertyid
ORDER BY b.CC_LIST_ID;