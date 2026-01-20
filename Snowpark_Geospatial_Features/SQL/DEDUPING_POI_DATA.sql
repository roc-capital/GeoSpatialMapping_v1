CREATE OR REPLACE TABLE SCRATCH.OSM_DATA.RAW_POIS_DEDUPED AS
SELECT *
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY LATITUDE, LONGITUDE, AMENITY
            ORDER BY CREATED_AT DESC
        ) as rn
    FROM SCRATCH.OSM_DATA.RAW_POIS
    WHERE AMENITY IN (
        'amenity:clinic', 'amenity:hospital', 'amenity:pharmacy', 'amenity:dentist',
        'amenity:veterinary', 'amenity:school', 'amenity:kindergarten', 'amenity:college',
        'amenity:university', 'amenity:cafe', 'amenity:restaurant', 'amenity:bar',
        'amenity:nightclub', 'amenity:pub', 'amenity:fuel', 'amenity:parking',
        'amenity:grave_yard', 'amenity:fire_station', 'amenity:place_of_worship',
        'amenity:funeral_home', 'amenity:library', 'amenity:theatre', 'amenity:bus_station',
        'shop:supermarket', 'shop:convenience', 'shop:grocery', 'shop:coffee',
        'shop:laundry', 'shop:variety_store', 'shop:discount', 'shop:pawnbroker', 'shop:alcohol',
        'tourism:museum', 'tourism:zoo',
        'leisure:garden', 'leisure:park', 'leisure:stadium', 'leisure:fitness_centre',
        'aeroway:aerodrome', 'aeroway:terminal',
        'railway:station', 'public_transport:station', 'public_transport:platform',
        'highway:bus_stop',
        'natural:beach', 'natural:wetland', 'natural:water', 'natural:peak',
        'waterway:riverbank',
        'landuse:cemetery', 'landuse:industrial'
    )
)
WHERE rn = 1;