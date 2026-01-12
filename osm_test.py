"""
EFFICIENT OSM AMENITY FEATURE EXTRACTION
Optimized for speed and minimal memory usage
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os

# -------------------------
# CONFIG
# -------------------------
INPUT_PARQUET = "/Users/jenny.lin/ImageDataParser/XGBoost_with_ImageData/XGBoost_Model_on_Basis_AVM_Data/data/inference_df.parquet"
OUTPUT_DIR = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/osm_features"

# Multiple Overpass servers (fallback)
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Core amenities only (most predictive for AVM)
AMENITIES = {
    'school': 'amenity=school',
    'hospital': 'amenity=hospital',
    'supermarket': 'shop=supermarket',
    'park': 'leisure=park',
}

AREAS = {
    'NC': 'area["ISO3166-2"="US-NC"]',
    'NY': 'area["ISO3166-2"="US-NY"]',
    'OH': 'area["ISO3166-2"="US-OH"]'
}

CACHE_FILE = f"{OUTPUT_DIR}/amenity_cache.json"
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds


# -------------------------
# CORE FUNCTIONS
# -------------------------

def query_osm(area_code: str, amenity_tag: str) -> list:
    """Query OSM with retry logic and fallback servers."""
    key, val = amenity_tag.split('=')
    query = f'[out:json][timeout:300];{AREAS[area_code]}->.a;nwr(area.a)["{key}"="{val}"];out center;'

    for attempt in range(MAX_RETRIES):
        for server_url in OVERPASS_SERVERS:
            try:
                print(f"    Attempt {attempt + 1}/{MAX_RETRIES} using {server_url.split('//')[1].split('/')[0]}...",
                      end=' ')

                r = requests.post(
                    server_url,
                    data={'data': query},
                    timeout=300,
                    headers={'User-Agent': 'AVM-Feature-Extraction/1.0'}
                )
                r.raise_for_status()

                coords = []
                for el in r.json().get('elements', []):
                    if el['type'] == 'node':
                        coords.append((el['lat'], el['lon']))
                    elif 'center' in el:
                        coords.append((el['center']['lat'], el['center']['lon']))

                print(f"✓ {len(coords)} found")
                return coords

            except requests.exceptions.Timeout:
                print(f"✗ Timeout")
                continue

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"✗ Rate limit")
                    if attempt < MAX_RETRIES - 1:
                        print(f"      Waiting {RETRY_DELAY}s before retry...")
                        time.sleep(RETRY_DELAY)
                    continue
                elif e.response.status_code == 504:
                    print(f"✗ Gateway timeout")
                    continue
                else:
                    print(f"✗ HTTP {e.response.status_code}")
                    continue

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                continue

    print(f"      Failed after {MAX_RETRIES} attempts across all servers")
    return []


def fetch_all_amenities():
    """Fetch all amenities sequentially to avoid rate limits."""

    # Try to load from cache
    if os.path.exists(CACHE_FILE):
        print(f"Loading from cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)

    print("Fetching amenities from OSM (sequential to avoid rate limits)...")
    amenities_data = {area: {} for area in AREAS.keys()}

    # Sequential requests (not parallel) to avoid rate limits
    total = len(AREAS) * len(AMENITIES)
    current = 0

    for area in AREAS.keys():
        print(f"\nQuerying {area}:")
        for name, tag in AMENITIES.items():
            current += 1
            print(f"  [{current}/{total}] {name}:")

            coords = query_osm(area, tag)
            amenities_data[area][name] = coords

            # Rate limiting between requests
            if coords:
                time.sleep(5)  # Successful query - wait 5 seconds
            else:
                time.sleep(10)  # Failed query - wait longer

    # Cache results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(amenities_data, f)
    print(f"\n✓ Cached results to: {CACHE_FILE}")

    return amenities_data


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine distance in miles."""
    lat1, lon1, lat2_arr, lon2_arr = map(np.radians, [lat1, lon1, lat2_arr, lon2_arr])
    dlat = lat2_arr - lat1
    dlon = lon2_arr - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def calc_features(lat, lon, amenity_coords):
    """Calculate proximity features for one property."""
    if not amenity_coords or pd.isna(lat) or pd.isna(lon):
        return np.nan, 0, 0

    # Convert to numpy arrays
    coords = np.array(amenity_coords)
    dists = haversine_vectorized(lat, lon, coords[:, 0], coords[:, 1])

    return dists.min(), (dists <= 1).sum(), (dists <= 3).sum()


def process_chunk(df_chunk, amenities_data, state_col='STATE'):
    """Process a chunk of properties."""
    features = []

    for _, row in df_chunk.iterrows():
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        state = row.get(state_col, 'NC')  # Default to NC if missing

        row_features = {}
        area_amenities = amenities_data.get(state, amenities_data.get('NC', {}))

        for amenity_name, coords in area_amenities.items():
            nearest, cnt_1mi, cnt_3mi = calc_features(lat, lon, coords)
            row_features[f'dist_{amenity_name}'] = nearest
            row_features[f'cnt1mi_{amenity_name}'] = cnt_1mi
            row_features[f'cnt3mi_{amenity_name}'] = cnt_3mi

        features.append(row_features)

    return pd.DataFrame(features)


def main():
    """Main execution."""
    print(f"\n{'=' * 60}")
    print("EFFICIENT OSM AMENITY EXTRACTION")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading property data...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} properties\n")

    # Fetch amenities (uses cache if available)
    amenities_data = fetch_all_amenities()

    # Summary
    print(f"\nAmenity summary:")
    for area, amenities in amenities_data.items():
        total = sum(len(coords) for coords in amenities.values())
        print(f"  {area}: {total:,} total locations")

    # Process in chunks for memory efficiency
    print(f"\nCreating proximity features...")
    chunk_size = 10000
    n_chunks = len(df) // chunk_size + 1

    feature_chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))

        chunk = df.iloc[start_idx:end_idx]
        features_df = process_chunk(chunk, amenities_data)
        feature_chunks.append(features_df)

        if (i + 1) % 10 == 0:
            print(f"  Processed {end_idx:,}/{len(df):,} properties")

    # Combine
    print("\nCombining results...")
    all_features = pd.concat(feature_chunks, ignore_index=True)
    result = pd.concat([df.reset_index(drop=True), all_features], axis=1)

    # Save
    output_path = f"{OUTPUT_DIR}/properties_with_osm_features.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"New features: {len(all_features.columns)}")
    print(f"Total columns: {len(result.columns)}")

    # Feature stats
    print(f"\nFeature statistics:")
    dist_cols = [c for c in all_features.columns if 'dist_' in c]
    print(all_features[dist_cols].describe().round(2))
    print()


if __name__ == "__main__":
    main()