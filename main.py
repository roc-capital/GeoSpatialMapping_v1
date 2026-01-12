import osmnx as ox
import pandas as pd
import numpy as np
import json
import os
import time

# -------------------------
# CONFIG
# -------------------------
INPUT_PARQUET = "/Users/jenny.lin/ImageDataParser/XGBoost_with_ImageData/XGBoost_Model_on_Basis_AVM_Data/data/inference_df.parquet"
OUTPUT_DIR = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/osm_features"
CACHE_FILE = f"{OUTPUT_DIR}/amenity_cache_cities.json"

# City centers with radius (in meters)
# Use 10-20km radius to keep queries manageable
CITIES = {
    'Charlotte_NC': {
        'center': (35.2271, -80.8431),  # Charlotte city center
        'radius': 15000,  # 15km = ~9 miles
    },
    'Raleigh_NC': {
        'center': (35.7796, -78.6382),  # Raleigh city center
        'radius': 15000,
    },
    'Columbus_OH': {
        'center': (39.9612, -82.9988),  # Columbus city center
        'radius': 15000,
    },
    'Minneapolis_MN': {
        'center': (44.9778, -93.2650),  # Minneapolis city center
        'radius': 15000,
    }
}

# Complete amenity list
AMENITIES = {
    # Healthcare
    'clinic': {'amenity': 'clinic'},
    'hospital': {'amenity': 'hospital'},
    'pharmacy': {'amenity': 'pharmacy'},
    'dentist': {'amenity': 'dentist'},
    'veterinary': {'amenity': 'veterinary'},

    # Education
    'school': {'amenity': 'school'},
    'kindergarten': {'amenity': 'kindergarten'},
    'college': {'amenity': 'college'},
    'university': {'amenity': 'university'},
    'library': {'amenity': 'library'},

    # Food & Drink
    'cafe': {'amenity': 'cafe'},
    'restaurant': {'amenity': 'restaurant'},
    'bar': {'amenity': 'bar'},
    'pub': {'amenity': 'pub'},
    'nightclub': {'amenity': 'nightclub'},

    # Shopping
    'supermarket': {'shop': 'supermarket'},
    'convenience': {'shop': 'convenience'},
    'grocery': {'shop': 'grocery'},
    'coffee_shop': {'shop': 'coffee'},
    'laundry': {'shop': 'laundry'},
    'variety_store': {'shop': 'variety_store'},
    'discount': {'shop': 'discount'},
    'pawnbroker': {'shop': 'pawnbroker'},
    'alcohol': {'shop': 'alcohol'},

    # Transportation
    'gas_station': {'amenity': 'fuel'},
    'parking': {'amenity': 'parking'},
    'bus_station': {'amenity': 'bus_station'},
    'bus_stop': {'highway': 'bus_stop'},

    # Death & Religion
    'cemetery': {'landuse': 'cemetery'},
    'grave_yard': {'amenity': 'grave_yard'},
    'fire_station': {'amenity': 'fire_station'},
    'place_of_worship': {'amenity': 'place_of_worship'},
    'funeral_home': {'amenity': 'funeral_home'},

    # Culture & Recreation
    'museum': {'tourism': 'museum'},
    'theatre': {'amenity': 'theatre'},
    'zoo': {'tourism': 'zoo'},
    'garden': {'leisure': 'garden'},
    'park': {'leisure': 'park'},
    'stadium': {'leisure': 'stadium'},
    'sports_center': {'leisure': 'sports_centre'},
    'fitness_centre': {'leisure': 'fitness_centre'},

    # Transportation Infrastructure
    'airport': {'aeroway': 'aerodrome'},
    'railway_station': {'railway': 'station'},

    # Natural Features
    'lake': {'natural': 'water', 'water': 'lake'},
    'beach': {'natural': 'beach'},
    'wetland': {'natural': 'wetland'},
    'peak': {'natural': 'peak'},

    # Land Use
    'industrial': {'landuse': 'industrial'},
}


# -------------------------
# FUNCTIONS
# -------------------------

def fetch_amenity_for_city(city_key, city_data, amenity_name, tags):
    """Fetch a single amenity type using radius around city center."""
    try:
        center = city_data['center']
        radius = city_data['radius']

        # OSMnx 2.0+ signature: (center_point, tags, dist)
        gdf = ox.features_from_point(
            center,  # tuple (lat, lon)
            tags,  # dict of tags
            radius  # distance in meters
        )

        # Extract coordinates (lat, lon)
        coords = []
        for idx, row in gdf.iterrows():
            geom = row.geometry

            if geom.geom_type == 'Point':
                coords.append((geom.y, geom.x))
            elif geom.geom_type in ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']:
                centroid = geom.centroid
                coords.append((centroid.y, centroid.x))

        return coords

    except Exception as e:
        return []


def fetch_city_amenities(city_key, city_data):
    """Fetch all amenities for a city."""
    print(f"\n{'=' * 60}")
    print(f"FETCHING: {city_key}")
    print(f"{'=' * 60}")
    print(f"Center: {city_data['center']}")
    print(f"Radius: {city_data['radius'] / 1000:.1f} km (~{city_data['radius'] / 1609:.1f} miles)\n")

    amenities_dict = {}
    total_found = 0

    for i, (amenity_name, tags) in enumerate(AMENITIES.items(), 1):
        print(f"[{i:2d}/{len(AMENITIES)}] {amenity_name:20s}...", end=' ')

        coords = fetch_amenity_for_city(city_key, city_data, amenity_name, tags)

        amenities_dict[amenity_name] = coords

        if coords:
            print(f"✓ {len(coords):6,}")
            total_found += len(coords)
        else:
            print(f"  (none)")

        # Small delay
        time.sleep(0.3)

    print(f"\n{city_key} TOTAL: {total_found:,} locations")

    return amenities_dict


def fetch_all_amenities():
    """Fetch amenities for all cities with caching."""

    # Try to load from cache
    if os.path.exists(CACHE_FILE):
        print(f"\n{'=' * 60}")
        print(f"LOADING FROM CACHE")
        print(f"{'=' * 60}")
        print(f"Cache: {CACHE_FILE}\n")

        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)

        for city, amenities in data.items():
            total = sum(len(coords) for coords in amenities.values())
            non_empty = sum(1 for coords in amenities.values() if len(coords) > 0)
            print(f"{city:20s}: {total:6,} locations across {non_empty:2d} types")

        return data

    # Fetch fresh data
    print(f"\n{'=' * 60}")
    print(f"FETCHING AMENITIES - CITY FOCUS")
    print(f"{'=' * 60}")
    print(f"Cities: {len(CITIES)}")
    print(f"Amenity types: {len(AMENITIES)}")
    print(f"This will take ~5-10 minutes...")

    all_data = {}

    for city_key, city_data in CITIES.items():
        amenities = fetch_city_amenities(city_key, city_data)
        all_data[city_key] = amenities

    # Save to cache
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(all_data, f)

    print(f"\n{'=' * 60}")
    print(f"✓ CACHED TO: {CACHE_FILE}")
    print(f"{'=' * 60}")

    return all_data


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine distance in miles."""
    lat1, lon1, lat2_arr, lon2_arr = map(np.radians, [lat1, lon1, lat2_arr, lon2_arr])
    dlat = lat2_arr - lat1
    dlon = lon2_arr - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def calc_features(lat, lon, amenity_coords):
    """Calculate proximity features."""
    if not amenity_coords or pd.isna(lat) or pd.isna(lon):
        return np.nan, 0, 0

    coords = np.array(amenity_coords)
    dists = haversine_vectorized(lat, lon, coords[:, 0], coords[:, 1])

    return dists.min(), (dists <= 1).sum(), (dists <= 3).sum()


def get_city_from_location(lat, lon):
    """Determine which city a property belongs to based on distance from center."""
    if pd.isna(lat) or pd.isna(lon):
        return None

    # Check which city center is closest and within radius
    for city_key, city_data in CITIES.items():
        center_lat, center_lon = city_data['center']
        radius_miles = city_data['radius'] / 1609.34  # Convert meters to miles

        # Calculate distance using haversine
        dist = haversine_vectorized(lat, lon,
                                    np.array([center_lat]),
                                    np.array([center_lon]))[0]

        if dist <= radius_miles:
            return city_key

    return None


def process_chunk(df_chunk, amenities_data):
    """Process a chunk of properties."""
    features = []

    for _, row in df_chunk.iterrows():
        lat, lon = row['LATITUDE'], row['LONGITUDE']

        # Determine which city this property is in
        city = get_city_from_location(lat, lon)

        row_features = {}

        if city and city in amenities_data:
            city_amenities = amenities_data[city]

            for amenity_name, coords in city_amenities.items():
                nearest, cnt_1mi, cnt_3mi = calc_features(lat, lon, coords)
                row_features[f'dist_{amenity_name}'] = nearest
                row_features[f'cnt1mi_{amenity_name}'] = cnt_1mi
                row_features[f'cnt3mi_{amenity_name}'] = cnt_3mi
        else:
            # Property not in any target city - set to NaN
            for amenity_name in AMENITIES.keys():
                row_features[f'dist_{amenity_name}'] = np.nan
                row_features[f'cnt1mi_{amenity_name}'] = 0
                row_features[f'cnt3mi_{amenity_name}'] = 0

        features.append(row_features)

    return pd.DataFrame(features)


def main():
    """Main execution."""
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print("CITY-FOCUSED OSM AMENITY EXTRACTION")
    print(f"{'=' * 60}\n")

    # Load property data
    print("Loading property data...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} properties")

    # Filter to properties in target cities
    print("\nFiltering to target cities...")
    df['target_city'] = df.apply(
        lambda row: get_city_from_location(row['LATITUDE'], row['LONGITUDE']),
        axis=1
    )

    df_cities = df[df['target_city'].notna()].copy()
    print(f"Properties in target cities: {len(df_cities):,}")

    # Show city distribution
    print("\nCity distribution:")
    for city in CITIES.keys():
        count = (df_cities['target_city'] == city).sum()
        pct = count / len(df_cities) * 100 if len(df_cities) > 0 else 0
        print(f"  {city:20s}: {count:8,} ({pct:5.1f}%)")

    # Fetch amenities
    amenities_data = fetch_all_amenities()

    # Create proximity features
    print(f"\n{'=' * 60}")
    print("CREATING PROXIMITY FEATURES")
    print(f"{'=' * 60}")
    print(f"Properties: {len(df_cities):,}")
    print(f"Features per property: {len(AMENITIES) * 3}\n")

    chunk_size = 10000
    n_chunks = len(df_cities) // chunk_size + 1

    feature_chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_cities))

        chunk = df_cities.iloc[start_idx:end_idx]
        features_df = process_chunk(chunk, amenities_data)
        feature_chunks.append(features_df)

        if (i + 1) % 5 == 0 or end_idx == len(df_cities):
            elapsed = time.time() - start_time
            rate = end_idx / elapsed if elapsed > 0 else 0
            remaining = (len(df_cities) - end_idx) / rate if rate > 0 else 0
            print(f"Processed {end_idx:,}/{len(df_cities):,} ({end_idx / len(df_cities) * 100:.1f}%) "
                  f"| {rate:.0f}/sec | ~{remaining / 60:.0f} min remaining")

    # Combine
    print("\nCombining results...")
    all_features = pd.concat(feature_chunks, ignore_index=True)
    result = pd.concat([df_cities.reset_index(drop=True), all_features], axis=1)

    # Save
    output_path = f"{OUTPUT_DIR}/properties_with_osm_cities.parquet"
    result.to_parquet(output_path, index=False)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Output: {output_path}")
    print(f"Properties processed: {len(result):,}")
    print(f"New features: {len(all_features.columns)}")
    print(f"Total columns: {len(result.columns)}")

    # Feature stats
    print(f"\n{'=' * 60}")
    print(f"SAMPLE FEATURE STATISTICS")
    print(f"{'=' * 60}")

    dist_cols = [c for c in all_features.columns if 'dist_' in c][:10]
    print(f"\nDistance to nearest (miles) - Top 10:")
    print(all_features[dist_cols].describe().round(2))

    print()


if __name__ == "__main__":
    try:
        import osmnx as ox
        import geopandas

        print(f"OSMnx: {ox.__version__}")
        print(f"GeoPandas: {geopandas.__version__}")
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nInstall: conda install -c conda-forge osmnx")
        exit(1)

    main()
