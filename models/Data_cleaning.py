################################################################################
# Stage 1: Data Cleaning Script
# Author: Vyshnavi Priya Kasarla
# Purpose: Clean and standardize NBA player datasets before integration
# Platform: PySpark on multi-node cluster (2+ VMs)
################################################################################

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, regexp_replace, when, count, isnan, mean
from pyspark.sql.types import IntegerType, FloatType

# ------------------------------------------------------------------------------
# 1. Initialize Spark session
# ------------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("NBA_Data_Cleaning") \
    .getOrCreate()

# ------------------------------------------------------------------------------
# 2. Load raw datasets
# ------------------------------------------------------------------------------
players_df = spark.read.csv("/home/sat3812/spark_project/Players.csv", header=True, inferSchema=True)
player_data_df = spark.read.csv("/home/sat3812/spark_project/player_data.csv", header=True, inferSchema=True)
seasons_df = spark.read.csv("/home/sat3812/spark_project/Seasons_Stats.csv", header=True, inferSchema=True)

print("✅ Raw Datasets Loaded")
print(f"Players: {players_df.count()} rows")
print(f"Player Data: {player_data_df.count()} rows")
print(f"Seasons: {seasons_df.count()} rows")

# ------------------------------------------------------------------------------
# 3. Trim spaces, unify case, clean special characters in column names
# ------------------------------------------------------------------------------
def clean_column_names(df):
    for old_name in df.columns:
        new_name = old_name.strip().lower().replace(' ', '_').replace('-', '_')
        df = df.withColumnRenamed(old_name, new_name)
    return df

players_df = clean_column_names(players_df)
player_data_df = clean_column_names(player_data_df)
seasons_df = clean_column_names(seasons_df)

# ------------------------------------------------------------------------------
# 4. Handle missing and invalid values
# ------------------------------------------------------------------------------
def clean_nulls(df):
    # Remove completely empty rows
    df = df.na.drop(how='all')
    
    # Replace placeholder strings like 'NA', 'NaN', '?', 'unknown' with None
    for c in df.columns:
        df = df.withColumn(c, when(col(c).isin('NA', 'NaN', 'na', '?', 'unknown', ''), None).otherwise(col(c)))
    return df

players_df = clean_nulls(players_df)
player_data_df = clean_nulls(player_data_df)
seasons_df = clean_nulls(seasons_df)

# ------------------------------------------------------------------------------
# 5. Convert numeric columns to proper data types
# ------------------------------------------------------------------------------
numeric_cols = [c for c, t in player_data_df.dtypes if t not in ['int', 'double', 'float']]

for c in numeric_cols:
    player_data_df = player_data_df.withColumn(c, regexp_replace(col(c), '[^0-9.-]', ''))
    player_data_df = player_data_df.withColumn(c, col(c).cast(FloatType()))

# ------------------------------------------------------------------------------
# 6. Handle missing numeric values using mean imputation
# ------------------------------------------------------------------------------
for c, t in player_data_df.dtypes:
    if t in ['int', 'double', 'float']:
        mean_val = player_data_df.select(mean(col(c))).collect()[0][0]
        if mean_val is not None:
            player_data_df = player_data_df.na.fill({c: round(mean_val, 2)})

# ------------------------------------------------------------------------------
# 7. Drop duplicate rows
# ------------------------------------------------------------------------------
players_df = players_df.dropDuplicates()
player_data_df = player_data_df.dropDuplicates()
seasons_df = seasons_df.dropDuplicates()

# ------------------------------------------------------------------------------
# 8. Sanity check – print schema and missing value counts
# ------------------------------------------------------------------------------
def print_missing(df, name):
    print(f"\n🧾 Missing Value Summary for {name}")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

print_missing(players_df, "Players")
print_missing(player_data_df, "Player Data")
print_missing(seasons_df, "Seasons")

# ------------------------------------------------------------------------------
# 9. Save cleaned outputs
# ------------------------------------------------------------------------------
players_df.write.mode("overwrite").option("header", True).csv("/home/sat3812/spark_project/cleaned_players")
player_data_df.write.mode("overwrite").option("header", True).csv("/home/sat3812/spark_project/cleaned_player_data")
seasons_df.write.mode("overwrite").option("header", True).csv("/home/sat3812/spark_project/cleaned_seasons")

print("\n✅ Data cleaning completed successfully!")
print("📂 Cleaned files saved in:")
print("   - cleaned_players/")
print("   - cleaned_player_data/")
print("   - cleaned_seasons/")

# ------------------------------------------------------------------------------
# 10. Stop Spark session
# ------------------------------------------------------------------------------
spark.stop()
################################################################################
