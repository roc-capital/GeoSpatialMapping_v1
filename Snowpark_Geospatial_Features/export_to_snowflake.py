#!/usr/bin/env python3
"""
Export PostgreSQL POI data directly to Snowflake
REQUIRES: Manual SSH tunnel running first
Run: ssh -i /Users/jenny.lin/Downloads/poi-scraper-key.pem -L 5433:localhost:5432 ubuntu@100.52.243.216 -N
"""
import psycopg2
import json
import os
from datetime import datetime
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

# PostgreSQL config (connects through SSH tunnel on localhost:5433)
PG_DB = {
    'dbname': 'poi_data',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5433
}

CONFIG_PATH = '/Users/jenny.lin/GeoSpatialMapping_Data_Scrap/config/config.json'


class PostgresToSnowflake:
    def __init__(self, config_path):
        print(f"Loading Snowflake config from: {config_path}")
        with open(config_path, 'r') as f:
            self.sf_config = json.load(f)

        print(f"✓ Config loaded")
        print(f"  Account: {self.sf_config.get('account')}")
        print(f"  Database: {self.sf_config.get('database')}")
        print(f"  Warehouse: {self.sf_config.get('warehouse')}")

        self.pg_conn = None
        self.sf_conn = None

    def connect_postgres(self):
        print(f"\nConnecting to PostgreSQL via SSH tunnel...")
        self.pg_conn = psycopg2.connect(**PG_DB)
        print("✓ PostgreSQL connected")

    def connect_snowflake(self):
        print(f"\nConnecting to Snowflake...")
        conn_params = {
            'account': self.sf_config['account'],
            'database': self.sf_config['database'],
            'warehouse': self.sf_config['warehouse'],
        }

        if self.sf_config.get('authenticator'):
            conn_params['authenticator'] = self.sf_config['authenticator']

        # Add workload_identity_provider if present
        if self.sf_config.get('workload_identity_provider'):
            conn_params['workload_identity_provider'] = self.sf_config['workload_identity_provider']

        import getpass
        if not self.sf_config.get('authenticator') or self.sf_config.get('authenticator') == 'WORKLOAD_IDENTITY':
            username = input("Snowflake Username (or press ENTER to skip): ")
            if username:
                password = getpass.getpass("Snowflake Password: ")
                conn_params['user'] = username
                conn_params['password'] = password

        self.sf_conn = snowflake.connector.connect(**conn_params)
        print("✓ Snowflake connected")

    def get_pg_stats(self):
        print("\n" + "=" * 70)
        print("POSTGRESQL DATABASE STATISTICS")
        print("=" * 70)

        cur = self.pg_conn.cursor()
        cur.execute("""
                    SELECT COUNT(*)                                            as total,
                           COUNT(DISTINCT state)                               as states,
                           COUNT(DISTINCT amenity_type)                        as amenities,
                           COUNT(DISTINCT (latitude, longitude, amenity_type)) as unique_triplets
                    FROM facilities
                    """)

        row = cur.fetchone()
        print(f"Total POIs: {row[0]:,}")
        print(f"Unique Triplets: {row[3]:,} (after deduplication)")
        print(f"Duplicates: {row[0] - row[3]:,}")
        print(f"States: {row[1]}")
        print(f"Amenity Types: {row[2]}")
        cur.close()
        print("=" * 70)
        return row[3]  # Return unique triplets count

    def export_to_dataframe(self, batch_size=50000):
        print("\nExporting from PostgreSQL...")

        # Query with deduplication - keep most recent entry for each triplet
        query = """
                SELECT DISTINCT \
                ON (latitude, longitude, amenity_type)
                    place_id,
                    name,
                    latitude,
                    longitude,
                    amenity_type as amenity,
                    state,
                    business_status,
                    rating,
                    user_ratings_total,
                    phone,
                    website,
                    address,
                    city,
                    zip_code,
                    created_at,
                    updated_at
                FROM facilities
                ORDER BY latitude, longitude, amenity_type, created_at DESC \
                """

        cur = self.pg_conn.cursor()
        cur.execute(query)

        columns = [
            'PLACE_ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'AMENITY',
            'STATE', 'BUSINESS_STATUS', 'RATING', 'USER_RATINGS_TOTAL',
            'PHONE', 'WEBSITE', 'ADDRESS', 'CITY', 'ZIP_CODE',
            'CREATED_AT', 'UPDATED_AT'
        ]

        all_data = []
        count = 0

        print(f"Fetching deduplicated data in batches of {batch_size:,}...")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            all_data.extend(rows)
            count += len(rows)
            print(f"  Progress: {count:,} rows fetched")

        cur.close()
        df = pd.DataFrame(all_data, columns=columns)

        # Clean and convert data types to avoid mismatches
        print("  Cleaning data types...")

        # Convert timestamps to strings
        df['CREATED_AT'] = df['CREATED_AT'].astype(str)
        df['UPDATED_AT'] = df['UPDATED_AT'].astype(str)

        # Convert numeric columns to proper types (handle NULLs)
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df['RATING'] = pd.to_numeric(df['RATING'], errors='coerce')
        df['USER_RATINGS_TOTAL'] = pd.to_numeric(df['USER_RATINGS_TOTAL'], errors='coerce').astype('Int64')

        # Ensure string columns are strings (not None)
        string_cols = ['PLACE_ID', 'NAME', 'AMENITY', 'STATE', 'BUSINESS_STATUS',
                       'PHONE', 'WEBSITE', 'ADDRESS', 'CITY', 'ZIP_CODE']
        for col in string_cols:
            df[col] = df[col].astype(str).replace('None', None)

        print(f"✓ Exported {len(df):,} deduplicated rows to DataFrame")
        return df

    def create_snowflake_table(self, schema='OSM_DATA'):
        print(f"\nCreating Snowflake table...")
        cur = self.sf_conn.cursor()

        cur.execute(f"CREATE SCHEMA IF NOT EXISTS SCRATCH.{schema}")
        cur.execute(f"USE SCHEMA SCRATCH.{schema}")

        ddl = f"""
        CREATE OR REPLACE TABLE SCRATCH.{schema}.RAW_POIS (
            PLACE_ID VARCHAR(100) PRIMARY KEY,
            NAME VARCHAR(1000),
            LATITUDE FLOAT,
            LONGITUDE FLOAT,
            AMENITY VARCHAR(200),
            STATE VARCHAR(10),
            BUSINESS_STATUS VARCHAR(100),
            RATING FLOAT,
            USER_RATINGS_TOTAL INTEGER,
            PHONE VARCHAR(100),
            WEBSITE VARCHAR(1000),
            ADDRESS VARCHAR(1000),
            CITY VARCHAR(200),
            ZIP_CODE VARCHAR(50),
            CREATED_AT VARCHAR(100),
            UPDATED_AT VARCHAR(100)
        )
        """

        cur.execute(ddl)
        print(f"✓ Table SCRATCH.{schema}.RAW_POIS created")
        cur.close()

    def load_to_snowflake(self, df, schema='OSM_DATA', table='RAW_POIS'):
        print(f"\nLoading {len(df):,} rows to Snowflake...")
        cur = self.sf_conn.cursor()
        cur.execute(f"USE SCHEMA SCRATCH.{schema}")

        success, nchunks, nrows, _ = write_pandas(
            conn=self.sf_conn,
            df=df,
            table_name=table,
            database='SCRATCH',
            schema=schema,
            auto_create_table=False,
            overwrite=True
        )

        if success:
            print(f"✓ Successfully loaded {nrows:,} rows in {nchunks} chunks")
        else:
            print(f"✗ Load failed")

        cur.close()
        return success

    def verify_snowflake_load(self, schema='OSM_DATA', table='RAW_POIS'):
        print(f"\nVerifying Snowflake data...")
        cur = self.sf_conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM SCRATCH.{schema}.{table}")
        total = cur.fetchone()[0]
        print(f"  Total rows: {total:,}")

        cur.execute(f"""
            SELECT STATE, COUNT(*) as cnt
            FROM SCRATCH.{schema}.{table}
            GROUP BY STATE ORDER BY cnt DESC LIMIT 5
        """)

        print(f"\n  Top 5 States:")
        for state, cnt in cur.fetchall():
            print(f"    {state}: {cnt:,}")

        cur.close()
        print("\n✓ Verification complete")

    def run(self):
        try:
            print("\n" + "=" * 70)
            print("POSTGRESQL → SNOWFLAKE EXPORT PIPELINE")
            print("=" * 70)

            self.connect_postgres()
            self.connect_snowflake()

            total_rows = self.get_pg_stats()
            df = self.export_to_dataframe()
            self.create_snowflake_table()
            success = self.load_to_snowflake(df)

            if success:
                self.verify_snowflake_load()
                print("\n" + "=" * 70)
                print("✓ EXPORT COMPLETE")
                print("=" * 70)
                print(f"\nData available in Snowflake:")
                print(f"  SCRATCH.OSM_DATA.RAW_POIS ({total_rows:,} rows)")
                print("=" * 70)
            else:
                print("\n✗ Export failed")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if self.pg_conn:
                self.pg_conn.close()
                print("\n✓ PostgreSQL closed")
            if self.sf_conn:
                self.sf_conn.close()
                print("✓ Snowflake closed")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SETUP: Make sure SSH tunnel is running first!")
    print("=" * 70)
    print("In another terminal, run:")
    print("  ssh -i /Users/jenny.lin/Downloads/poi-scraper-key.pem \\")
    print("      -L 5433:localhost:5432 ubuntu@100.52.243.216 -N")
    print("=" * 70)

    input("\nPress ENTER when SSH tunnel is ready...")

    if not os.path.exists(CONFIG_PATH):
        print(f"✗ Config not found: {CONFIG_PATH}")
        exit(1)

    exporter = PostgresToSnowflake(CONFIG_PATH)
    exporter.run()