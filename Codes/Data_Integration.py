# Data Integration 
from pyspark.sql import SparkSession, functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_palette("muted")
fig_dir = "integration_figures"
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("NBADataIntegration").getOrCreate()

# Load cleaned datasets
players_df  = spark.read.csv("clean_output/clean_players_csv", header=True, inferSchema=True)
player_data = spark.read.csv("clean_output/clean_player_data", header=True, inferSchema=True)
seasons_df  = spark.read.csv("clean_output/clean_seasons_stats", header=True, inferSchema=True)

# Normalize player names
normalize = F.udf(lambda s: s.lower().strip() if s else None)
players_df  = players_df.withColumn("name_norm", normalize(F.col("Player"))) if "Player" in players_df.columns else players_df
player_data = player_data.withColumn("name_norm", normalize(F.col("name"))) if "name" in player_data.columns else player_data
seasons_df  = seasons_df.withColumn("name_norm", normalize(F.col("Player"))) if "Player" in seasons_df.columns else seasons_df

# Rename overlapping columns to avoid ambiguity
player_data = player_data.select([F.col(c).alias(f"pd_{c}") if c != "name_norm" else F.col(c) for c in player_data.columns])
players_df  = players_df.select([F.col(c).alias(f"pl_{c}") if c != "name_norm" else F.col(c) for c in players_df.columns])
seasons_df  = seasons_df.select([F.col(c).alias(f"ss_{c}") if c != "name_norm" else F.col(c) for c in seasons_df.columns])

# Join datasets 
roster = players_df.join(player_data, on="name_norm", how="outer")
integrated_df = seasons_df.join(roster, on="name_norm", how="left")

# Drop duplicate 
duplicate_cols = [c for c in integrated_df.columns if integrated_df.columns.count(c) > 1]
integrated_df = integrated_df.drop(*duplicate_cols)

# Correlation analysis to remove redundancy
num_cols = [c for c, t in integrated_df.dtypes if t in ("double", "float", "int", "bigint")]
safe_cols = [c for c in num_cols if c not in ("name_norm",)] 

if safe_cols:
    # Drop columns with all nulls or constant values
    for c in safe_cols:
        if integrated_df.select(c).distinct().count() <= 1:
            integrated_df = integrated_df.drop(c)
    # Collect correlation
    sample_pdf = integrated_df.select(safe_cols).sample(fraction=0.2, seed=42).toPandas()
    corr = sample_pdf.corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i,j] > 0.85:
                to_drop.add(corr.columns[j])
    integrated_df = integrated_df.drop(*to_drop)

    # Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap – Integrated NBA Data")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/correlation_heatmap.png")
    plt.close()
else:
    print(" No numeric columns available for correlation.")
integrated_df.write.mode("overwrite").csv("clean_output/integrated_nba_stats", header=True)

print("Integration Complete")
print(f"Rows: {integrated_df.count()}")
print(f"Numeric columns analyzed: {len(num_cols)}")
print(f"Dropped due to correlation: {len(to_drop) if safe_cols else 0}")
print(f"Visuals saved in: {fig_dir}")

spark.stop()
