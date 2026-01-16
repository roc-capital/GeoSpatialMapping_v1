WITH props AS (
  SELECT COUNT(DISTINCT CC_PROPERTY_ID) AS total_properties
  FROM SCRATCH.OSM_DATA.GSF_SCREENING
  WHERE CC_PROPERTY_ID IS NOT NULL
),
amenity_props AS (
  SELECT
    AMENITY,
    COUNT(DISTINCT CC_PROPERTY_ID) AS properties_with_amenity
  FROM SCRATCH.OSM_DATA.GSF_SCREENING
  WHERE CC_PROPERTY_ID IS NOT NULL
    AND AMENITY IS NOT NULL
  GROUP BY AMENITY
)
SELECT
  a.AMENITY,
  a.properties_with_amenity,
  p.total_properties,
  a.properties_with_amenity * 1.0 / p.total_properties AS coverage_rate
FROM amenity_props a
CROSS JOIN props p
ORDER BY coverage_rate DESC, properties_with_amenity DESC, AMENITY;