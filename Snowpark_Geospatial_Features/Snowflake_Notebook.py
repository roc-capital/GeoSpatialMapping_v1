import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import (
    col, count, count_distinct, avg, stddev, min as min_, max as max_,
    corr, sum as sum_, lit, abs as abs_, coalesce, when, log
)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


# ============================================================================
# FEATURE SCREENER CLASS
# ============================================================================

class FeatureScreener:
    def __init__(self, session, schema="SCRATCH.OSM_DATA"):
        self.session = session
        self.schema = schema
        self.log = {'stages': []}

    def log_stage(self, stage_name, input_count, output_count, details=None):
        entry = {
            'stage': stage_name,
            'input': input_count,
            'output': output_count,
            'dropped': input_count - output_count
        }
        if details:
            entry.update(details)
        self.log['stages'].append(entry)
        print(f"{stage_name}: {input_count} → {output_count}")

    def screen_amenities(self, source_table, distance_col='NEAREST_DIST_METERS',
                         min_coverage=0.10, min_cv=0.15, max_avg_distance=10000):
        """Screen amenities using SQL"""
        print("\n=== STAGE 1: AMENITY SCREENING ===")

        self.session.sql(f"""
            CREATE OR REPLACE TABLE {self.schema}.SCREENED_AMENITIES AS
            WITH props AS (
              SELECT COUNT(DISTINCT CC_PROPERTY_ID) AS total_properties
              FROM {source_table}
              WHERE CC_PROPERTY_ID IS NOT NULL
            ),
            amenity_stats AS (
              SELECT
                AMENITY,
                COUNT(DISTINCT CC_PROPERTY_ID) AS properties_with_amenity,
                AVG({distance_col}) as avg_distance,
                STDDEV({distance_col}) as std_distance
              FROM {source_table}
              WHERE CC_PROPERTY_ID IS NOT NULL AND AMENITY IS NOT NULL
              GROUP BY AMENITY
            )
            SELECT
              a.AMENITY,
              a.properties_with_amenity,
              p.total_properties,
              a.properties_with_amenity * 1.0 / p.total_properties AS coverage_rate,
              a.avg_distance,
              a.std_distance,
              a.std_distance / NULLIF(a.avg_distance, 0) AS cv
            FROM amenity_stats a
            CROSS JOIN props p
            WHERE a.properties_with_amenity * 1.0 / p.total_properties >= {min_coverage}
              AND a.std_distance / NULLIF(a.avg_distance, 0) >= {min_cv}
              AND a.avg_distance <= {max_avg_distance}
            ORDER BY coverage_rate DESC
        """).collect()

        amenities_df = self.session.table(f"{self.schema}.SCREENED_AMENITIES")
        amenity_list = [row['AMENITY'] for row in amenities_df.select("AMENITY").collect()]

        total = self.session.sql(f"""
            SELECT COUNT(DISTINCT AMENITY) as cnt FROM {source_table} WHERE AMENITY IS NOT NULL
        """).collect()[0]['CNT']

        self.log_stage("Amenity Screening", total, len(amenity_list))

        print(f"\n  Surviving Amenities ({len(amenity_list)}):")
        for i, amenity in enumerate(amenity_list[:10], 1):
            print(f"    {i}. {amenity}")
        if len(amenity_list) > 10:
            print(f"    ... and {len(amenity_list) - 10} more")

        return amenity_list

    def engineer_features(self, source_table, amenity_list,
                          target_col='CURRENTSALESPRICE',
                          distance_col='NEAREST_DIST_METERS'):
        """Create features using SQL"""
        print("\n=== STAGE 2: FEATURE ENGINEERING ===")

        feature_sqls = []
        for amenity in amenity_list:
            safe_name = amenity.upper().replace(' ', '_').replace('-', '_').replace('.', '_').replace("'", '').replace(
                '(', '').replace(')', '')
            sql_safe_amenity = amenity.replace("'", "''")

            # Binary: has amenity within 5km (5000 meters)
            feature_sqls.append(
                f"MAX(CASE WHEN AMENITY = '{sql_safe_amenity}' AND {distance_col} <= 5000 THEN 1 ELSE 0 END) AS HAS_{safe_name}")

            # Distance to nearest
            feature_sqls.append(
                f"MIN(CASE WHEN AMENITY = '{sql_safe_amenity}' THEN {distance_col} END) AS DIST_{safe_name}")

            # Count within 2km
            feature_sqls.append(
                f"SUM(CASE WHEN AMENITY = '{sql_safe_amenity}' AND {distance_col} <= 2000 THEN 1 ELSE 0 END) AS COUNT_{safe_name}_2KM")

        features_sql = ",\n              ".join(feature_sqls)

        print(f"  Creating features for {len(amenity_list)} amenities...")

        # **FILTER OUTLIERS - REMOVE TOP/BOTTOM 1% OF PRICES**
        self.session.sql(f"""
            CREATE OR REPLACE TABLE {self.schema}.FEATURES_ENGINEERED AS
            WITH price_bounds AS (
                SELECT 
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY {target_col}) as lower_bound,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {target_col}) as upper_bound
                FROM (
                    SELECT DISTINCT CC_PROPERTY_ID, {target_col} 
                    FROM {source_table} 
                    WHERE CC_PROPERTY_ID IS NOT NULL 
                      AND {target_col} IS NOT NULL
                      AND {target_col} > 0
                )
            )
            SELECT p.CC_PROPERTY_ID, p.{target_col}, {features_sql}
            FROM (
                SELECT DISTINCT CC_PROPERTY_ID, {target_col} 
                FROM {source_table} 
                WHERE CC_PROPERTY_ID IS NOT NULL
                  AND {target_col} IS NOT NULL
                  AND {target_col} > 0
                  AND {target_col} BETWEEN (SELECT lower_bound FROM price_bounds) 
                                       AND (SELECT upper_bound FROM price_bounds)
            ) p
            LEFT JOIN {source_table} g ON p.CC_PROPERTY_ID = g.CC_PROPERTY_ID
            GROUP BY p.CC_PROPERTY_ID, p.{target_col}
        """).collect()

        df = self.session.table(f"{self.schema}.FEATURES_ENGINEERED")
        feature_cols = [c for c in df.columns if c not in ['CC_PROPERTY_ID', target_col]]

        # **PRINT PRICE DIAGNOSTICS**
        price_stats = df.select(
            count(col(target_col)).alias('count'),
            avg(col(target_col)).alias('avg'),
            stddev(col(target_col)).alias('stddev'),
            min_(col(target_col)).alias('min'),
            max_(col(target_col)).alias('max')
        ).collect()[0]

        print(f"\n  Price Statistics (after outlier removal):")
        print(f"    Properties: {price_stats['COUNT']}")
        print(f"    Average: ${price_stats['AVG']:,.0f}")
        print(f"    Std Dev: ${price_stats['STDDEV']:,.0f}")
        print(f"    Min: ${price_stats['MIN']:,.0f}")
        print(f"    Max: ${price_stats['MAX']:,.0f}")

        self.log_stage("Feature Engineering", len(amenity_list), len(feature_cols))

        print(f"  Created {len(feature_cols)} features")
        print(f"    - {len([f for f in feature_cols if f.startswith('HAS_')])} binary (HAS_) features")
        print(f"    - {len([f for f in feature_cols if f.startswith('DIST_')])} distance (DIST_) features")
        print(f"    - {len([f for f in feature_cols if f.startswith('COUNT_')])} count (COUNT_) features")

        return df, feature_cols

    def analyze_price_distance(self, df, feature_cols, target_col='CURRENTSALESPRICE'):
        """Analyze price-distance relationships"""
        print("\n=== PRICE-DISTANCE ANALYSIS ===")

        dist_features = [f for f in feature_cols if f.startswith('DIST_')]

        if not dist_features:
            print("  No distance features found")
            return pd.DataFrame()

        # Limit to 20 for performance
        dist_features = dist_features[:20]

        print(f"  Analyzing {len(dist_features)} distance features...")

        corr_exprs = [corr(col(f), col(target_col)).alias(f) for f in dist_features]
        corrs_row = df.select(corr_exprs).collect()[0]

        results = []
        for feat in dist_features:
            corr_val = corrs_row[feat]
            if corr_val is None:
                continue

            amenity_name = feat.replace('DIST_', '')
            results.append({
                'feature': feat,
                'amenity': amenity_name,
                'correlation': corr_val,
                'direction': 'Closer=Higher' if corr_val < -0.05 else 'Farther=Higher' if corr_val > 0.05 else 'Neutral',
                'strength': 'Strong' if abs(corr_val) > 0.3 else 'Moderate' if abs(corr_val) > 0.15 else 'Weak'
            })

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results).sort_values('correlation', key=lambda x: abs(x), ascending=False)

        print("\n  Top 10 Price-Distance Relationships:")
        print("  " + "-" * 68)
        for _, row in results_df.head(10).iterrows():
            symbol = "↓" if row['correlation'] < 0 else "↑"
            print(f"  {row['amenity']:28} | r={row['correlation']:+.3f} {symbol} | {row['strength']:10}")

        return results_df

    def filter_by_quality(self, df, feature_cols, target_col='CURRENTSALESPRICE',
                          min_coverage=0.50, min_std=0.01):
        """Filter by coverage and variance"""
        print("\n=== STAGE 3: QUALITY FILTER ===")

        total_rows = df.count()
        survivors = []

        print(f"  Checking {len(feature_cols)} features...")

        for i, feat in enumerate(feature_cols):
            if i > 0 and i % 20 == 0:
                print(f"    Progress: {i}/{len(feature_cols)}")

            stats = df.select(
                count(col(feat)).alias('non_null'),
                stddev(col(feat)).alias('std_val')
            ).collect()[0]

            coverage = stats['NON_NULL'] / total_rows if total_rows > 0 else 0
            std_val = stats['STD_VAL'] or 0

            if coverage >= min_coverage and std_val >= min_std:
                survivors.append(feat)

        self.log_stage("Quality Filter", len(feature_cols), len(survivors))
        return df.select('CC_PROPERTY_ID', target_col, *survivors), survivors

    def screen_by_correlation(self, df, feature_cols, target_col='CURRENTSALESPRICE',
                              min_target_corr=0.02, max_feature_corr=0.85):
        """Correlation screening"""
        print("\n=== STAGE 4: CORRELATION SCREENING ===")

        # Target correlations
        print("  Computing target correlations...")
        corr_exprs = [corr(col(f), col(target_col)).alias(f) for f in feature_cols]
        target_corrs = df.select(corr_exprs).collect()[0].as_dict()

        # **DIAGNOSTIC OUTPUT**
        print(f"\n  DIAGNOSTIC: Correlation Distribution:")
        corr_values = [abs(v) for v in target_corrs.values() if v is not None]
        if corr_values:
            print(f"    Max correlation: {max(corr_values):.4f}")
            print(f"    Min correlation: {min(corr_values):.4f}")
            print(f"    Avg correlation: {sum(corr_values) / len(corr_values):.4f}")
            print(f"    Features above {min_target_corr}: {sum(1 for c in corr_values if c >= min_target_corr)}")
        else:
            print(f"    WARNING: All correlations are NULL!")

        survivors_data = []
        for f in feature_cols:
            corr_val = target_corrs.get(f.upper())
            if corr_val is not None and abs(corr_val) >= min_target_corr:
                survivors_data.append({'feature': f, 'corr': corr_val, 'abs_corr': abs(corr_val)})

        # **HANDLE EMPTY SURVIVORS**
        if not survivors_data:
            print(f"  WARNING: No features met min_target_corr={min_target_corr}")
            print(f"  Keeping all {len(feature_cols)} features")
            survivors_df = pd.DataFrame([
                {'feature': f, 'corr': target_corrs.get(f.upper(), 0), 'abs_corr': abs(target_corrs.get(f.upper(), 0))}
                for f in feature_cols
            ]).sort_values('abs_corr', ascending=False)
            survivors = survivors_df['feature'].tolist()
        else:
            survivors_df = pd.DataFrame(survivors_data).sort_values('abs_corr', ascending=False)
            survivors = survivors_df['feature'].tolist()

        print(f"  After target filter: {len(feature_cols)} → {len(survivors)}")

        # Multicollinearity removal
        print("  Removing multicollinear features...")
        feature_scores = dict(zip(survivors_df['feature'], survivors_df['abs_corr']))
        to_drop = set()

        for i, feat_i in enumerate(survivors):
            if feat_i in to_drop:
                continue

            if i % 10 == 0:
                print(f"    Progress: {i}/{len(survivors)}")

            remaining = [f for f in survivors[i + 1:] if f not in to_drop]
            if not remaining:
                continue

            # Process in batches
            for j in range(0, len(remaining), 15):
                batch = remaining[j:j + 15]
                pair_corrs = df.select([
                    corr(col(feat_i), col(feat_j)).alias(feat_j)
                    for feat_j in batch
                ]).collect()[0].as_dict()

                for feat_j, corr_val in pair_corrs.items():
                    if corr_val is not None and abs(corr_val) > max_feature_corr:
                        if feature_scores[feat_i] >= feature_scores[feat_j]:
                            to_drop.add(feat_j)
                        else:
                            to_drop.add(feat_i)
                            break

                if feat_i in to_drop:
                    break

        final_survivors = [f for f in survivors if f not in to_drop]
        self.log_stage("Correlation Screening", len(feature_cols), len(final_survivors), {
            'after_target': len(survivors),
            'multicollinear_dropped': len(to_drop)
        })

        return df.select('CC_PROPERTY_ID', target_col, *final_survivors), final_survivors, survivors_df

    def create_composites(self, df, survivors, corr_df, target_col='CURRENTSALESPRICE'):
        """Create composite features"""
        print("\n=== STAGE 5: COMPOSITE CREATION ===")

        groups = {
            'ESSENTIAL': ['HOSPITAL', 'CLINIC', 'PHARMACY', 'FIRE_STATION', 'POLICE', 'DENTIST'],
            'EDUCATION': ['SCHOOL', 'KINDERGARTEN', 'COLLEGE', 'UNIVERSITY', 'LIBRARY'],
            'CULTURAL': ['MUSEUM', 'THEATRE', 'CINEMA', 'GALLERY', 'ARTS'],
            'DAILY': ['RESTAURANT', 'CAFE', 'GROCERY', 'SUPERMARKET', 'BANK', 'POST', 'FUEL'],
            'RECREATION': ['PARK', 'PLAYGROUND', 'GYM', 'SPORTS', 'POOL']
        }

        feature_scores = dict(zip(corr_df['feature'], corr_df['abs_corr']))
        composites_added = []
        components_removed = set()

        for group_name, keywords in groups.items():
            has_features = [
                f for f in survivors
                if f.startswith('HAS_') and any(kw in f.upper() for kw in keywords)
            ]

            if len(has_features) < 2:
                continue

            comp_name = f"{group_name}_SCORE"

            # Build sum expression properly
            comp_expr = coalesce(col(has_features[0]), lit(0))
            for feat in has_features[1:]:
                comp_expr = comp_expr + coalesce(col(feat), lit(0))

            df = df.with_column(comp_name, comp_expr)

            comp_corr_val = df.select(corr(col(comp_name), col(target_col))).collect()[0][0]
            if comp_corr_val is None:
                continue

            best_component = max([feature_scores.get(f, 0) for f in has_features])
            improvement = abs(comp_corr_val) - best_component

            print(
                f"  {comp_name}: r={comp_corr_val:.3f}, improvement={improvement:+.3f} ({len(has_features)} components)")

            if improvement > 0.02:
                composites_added.append(comp_name)
                components_removed.update(has_features)

        final_survivors = [f for f in survivors if f not in components_removed]
        final_survivors.extend(composites_added)

        self.log_stage("Composite Creation", len(survivors), len(final_survivors), {
            'composites_added': len(composites_added),
            'components_removed': len(components_removed)
        })

        return df, final_survivors

    def model_based_ranking(self, df, feature_cols, target_col='CURRENTSALESPRICE',
                            sample_size=5000, top_n=25):
        """Model-based ranking using Random Forest"""
        print("\n=== STAGE 6: MODEL-BASED RANKING ===")

        print(f"  Sampling {sample_size} rows...")
        df_sample = df.sample(n=min(sample_size, df.count()))
        df_pd = df_sample.select(feature_cols + [target_col]).to_pandas()

        # Handle missing values
        print(f"  Cleaning data...")
        df_pd = df_pd.fillna(df_pd.median(numeric_only=True))

        # Remove rows where target is null
        df_pd = df_pd.dropna(subset=[target_col])

        if len(df_pd) < 100:
            print("  Warning: Too few valid samples after cleaning")
            return df, feature_cols[:top_n], pd.DataFrame()

        X = df_pd[feature_cols]
        y = df_pd[target_col]

        print(f"  Training Random Forest on {len(df_pd)} samples...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(top_n)['feature'].tolist()

        print(f"\n  Top 10 Features by Importance:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:40} {row['importance']:.4f}")

        self.log_stage("Model Ranking", len(feature_cols), len(top_features))

        return df.select('CC_PROPERTY_ID', target_col, *top_features), top_features, importance_df

    def save_reports(self, features, importance_df, price_dist_df):
        """Save comprehensive reports to tables"""
        from datetime import datetime

        print("\n" + "=" * 70)
        print("SAVING DETAILED REPORTS")
        print("=" * 70)

        # Report 1: Amenities
        print(f"  ✓ {self.schema}.SCREENED_AMENITIES (already saved)")

        # Report 2: Pipeline Summary
        summary_data = []
        for stage in self.log['stages']:
            summary_data.append({
                'stage': stage['stage'],
                'input_count': stage['input'],
                'output_count': stage['output'],
                'dropped_count': stage['dropped'],
                'retention_pct': round(stage['output'] / stage['input'] * 100, 1) if stage['input'] > 0 else 0
            })

        summary_df = self.session.create_dataframe(summary_data)
        summary_df.write.mode("overwrite").save_as_table(
            f"{self.schema}.REPORT_PIPELINE_SUMMARY"
        )
        print(f"  ✓ {self.schema}.REPORT_PIPELINE_SUMMARY")

        # Report 2B: Detailed Output Log with Timestamps
        # **FIX: Drop existing table first to avoid schema mismatch**
        try:
            self.session.sql(f"DROP TABLE IF EXISTS {self.schema}.PIPELINE_OUTPUT_LOG").collect()
        except:
            pass

        log_data = []
        for stage in self.log['stages']:
            log_data.append({
                'run_timestamp': datetime.now(),
                'stage_name': stage['stage'],
                'log_message': f"{stage['stage']}: {stage['input']} → {stage['output']} (-{stage['dropped']})",
                'input_count': stage['input'],
                'output_count': stage['output'],
                'dropped_count': stage['dropped'],
                'retention_pct': round(stage['output'] / stage['input'] * 100, 1) if stage['input'] > 0 else 0
            })

        log_df = self.session.create_dataframe(log_data)
        log_df.write.mode("overwrite").save_as_table(  # **CHANGED TO OVERWRITE**
            f"{self.schema}.PIPELINE_OUTPUT_LOG"
        )
        print(f"  ✓ {self.schema}.PIPELINE_OUTPUT_LOG")

        # Report 3: Final Features with Importance
        if not importance_df.empty:
            features_data = []
            for i, feat in enumerate(features, 1):
                imp_vals = importance_df[importance_df['feature'] == feat]['importance'].values
                features_data.append({
                    'rank': i,
                    'feature_name': feat,
                    'importance': float(imp_vals[0]) if len(imp_vals) > 0 else 0.0,
                    'feature_type': 'Distance' if feat.startswith('DIST_')
                    else 'Binary' if feat.startswith('HAS_')
                    else 'Count' if feat.startswith('COUNT_')
                    else 'Composite'
                })

            features_df = self.session.create_dataframe(features_data)
            features_df.write.mode("overwrite").save_as_table(
                f"{self.schema}.REPORT_FINAL_FEATURES"
            )
            print(f"  ✓ {self.schema}.REPORT_FINAL_FEATURES")

        # Report 4: Price-Distance Analysis
        if not price_dist_df.empty:
            dist_data = []
            for _, row in price_dist_df.iterrows():
                dist_data.append({
                    'amenity': row['amenity'],
                    'correlation': float(row['correlation']),
                    'abs_correlation': abs(float(row['correlation'])),
                    'direction': row['direction'],
                    'strength': row['strength']
                })

            dist_df = self.session.create_dataframe(dist_data)
            dist_df.write.mode("overwrite").save_as_table(
                f"{self.schema}.REPORT_PRICE_DISTANCE"
            )
            print(f"  ✓ {self.schema}.REPORT_PRICE_DISTANCE")

        print("\n" + "=" * 70)
        print("ALL REPORTS SAVED")
        print("=" * 70)
        print("\nQuery these tables to view results:")
        print(f"  • SELECT * FROM {self.schema}.REPORT_PIPELINE_SUMMARY;")
        print(f"  • SELECT * FROM {self.schema}.PIPELINE_OUTPUT_LOG ORDER BY run_timestamp DESC;")
        print(f"  • SELECT * FROM {self.schema}.REPORT_FINAL_FEATURES ORDER BY rank;")
        print(f"  • SELECT * FROM {self.schema}.REPORT_PRICE_DISTANCE ORDER BY abs_correlation DESC;")
        print(f"  • SELECT * FROM {self.schema}.SCREENED_AMENITIES ORDER BY coverage_rate DESC;")

    def run_full_pipeline(self, source_table,
                          target_col='CURRENTSALESPRICE',
                          distance_col='NEAREST_DIST_METERS',
                          final_n=20):
        """Execute complete pipeline"""
        print("\n" + "=" * 70)
        print("GEOSPATIAL FEATURE SCREENING PIPELINE")
        print("=" * 70)
        print(f"Source Table: {source_table}")
        print(f"Target Column: {target_col}")
        print(f"Distance Column: {distance_col}")
        print(f"Final Features Target: {final_n}")
        print("=" * 70)

        # Stage 1: Screen amenities
        amenities = self.screen_amenities(source_table, distance_col)

        # Stage 2: Engineer features
        df, features = self.engineer_features(source_table, amenities, target_col, distance_col)

        # Stage 2.5: Analyze price-distance
        price_dist_analysis = self.analyze_price_distance(df, features, target_col)

        # Stage 3: Quality filter
        df, features = self.filter_by_quality(df, features, target_col)

        # Stage 4: Correlation screening (lowered min_target_corr to 0.02)
        df, features, corr_df = self.screen_by_correlation(df, features, target_col, min_target_corr=0.02)

        # Stage 5: Composites
        df, features = self.create_composites(df, features, corr_df, target_col)

        # Stage 6: Model ranking (final selection)
        df, features, importance_df = self.model_based_ranking(
            df, features, target_col,
            top_n=final_n
        )

        # Save final dataset
        print(f"\n  Saving final dataset...")
        df.write.mode("overwrite").save_as_table(f"{self.schema}.FEATURES_FINAL")

        # Save all reports
        self.save_reports(features, importance_df, price_dist_analysis)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nFinal Features ({len(features)}):")
        for i, feat in enumerate(features, 1):
            if not importance_df.empty:
                imp = importance_df[importance_df['feature'] == feat]['importance'].values
                imp_val = imp[0] if len(imp) > 0 else 0
                print(f"  {i:2d}. {feat:40} (importance: {imp_val:.4f})")
            else:
                print(f"  {i:2d}. {feat}")

        print(f"\nOutput Tables:")
        print(f"  • {self.schema}.FEATURES_FINAL ← Use this for GP-DAG modeling")
        print(f"  • {self.schema}.FEATURES_ENGINEERED (all features before selection)")
        print(f"  • {self.schema}.SCREENED_AMENITIES (amenity statistics)")

        print(f"\nPipeline Summary:")
        for stage in self.log['stages']:
            print(f"  {stage['stage']:30} {stage['input']:4d} → {stage['output']:4d} (-{stage['dropped']:3d})")

        return df, features


# ============================================================================
# MAIN HANDLER
# ============================================================================

def main(session: snowpark.Session):
    """Main handler for Python Worksheet"""

    print("=" * 70)
    print("STARTING GEOSPATIAL FEATURE SCREENING")
    print("=" * 70)
    print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")
    print(f"Warehouse: {session.get_current_warehouse()}")
    print("=" * 70)

    # Initialize screener
    screener = FeatureScreener(session, schema="SCRATCH.OSM_DATA")

    # Run pipeline with your column names
    df_final, final_features = screener.run_full_pipeline(
        source_table="SCRATCH.OSM_DATA.GSF_SCREENING",
        target_col="CURRENTSALESPRICE",
        distance_col="NEAREST_DIST_METERS",
        final_n=20
    )

    # Return preview DataFrame (required for Python Worksheets)
    print("\n" + "=" * 70)
    print("RETURNING PREVIEW")
    print("=" * 70)
    print("Returning first 100 rows of final dataset...")
    print("For full dataset, query: SCRATCH.OSM_DATA.FEATURES_FINAL")
    print("=" * 70)

    return df_final.limit(100)