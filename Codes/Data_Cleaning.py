# Data Cleaning
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_palette("pastel")

# Paths
players_path       = "Players.csv"
player_data_path   = "player_data.csv"
seasons_path       = "Seasons_Stats.csv"
clean_players_dir  = "clean_output/clean_players_csv"
clean_player_data_dir = "clean_output/clean_player_data"
clean_seasons_dir  = "clean_output/clean_seasons_stats"
figures_dir        = "cleaning_figures"
pathlib.Path(figures_dir).mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("CleanNBAData").getOrCreate()

# Load and deduplicate
players_df  = spark.read.csv(players_path, header=True, inferSchema=True).dropDuplicates()
player_data = spark.read.csv(player_data_path, header=True, inferSchema=True).dropDuplicates()
seasons_df  = spark.read.csv(seasons_path, header=True, inferSchema=True).dropDuplicates()

# Drop empty columns in seasons
for col in ["blanl","blank2"]:
    if col in seasons_df.columns:
        seasons_df = seasons_df.drop(col)

# Height conversion function
def convert_height(h):
    if h is None: return None
    try:
        if isinstance(h, str) and '-' in h:
            feet, inches = h.split('-')
            if feet.isdigit() and inches.isdigit():
                return int(feet)*12 + int(inches)
        return int(round(float(h)/2.54))
    except Exception:
        return None

convert_height_udf = F.udf(convert_height, IntegerType())

# -------- Clean player_data --------
# Fill string columns except height
for c, t in player_data.dtypes:
    if t == "string" and c != "height":
        player_data = player_data.fillna({c: "Unknown"})
# Convert height and impute missing heights with mean
player_data = player_data.withColumn("height_in", convert_height_udf(F.col("height")))
mean_height_pd = player_data.agg(F.avg("height_in")).first()[0]
player_data = player_data.withColumn("height_in",
                                     F.when(F.col("height_in").isNull(), mean_height_pd)
                                       .otherwise(F.col("height_in")))
# Impute all numeric columns (including weight, year_start, year_end) with their mean
num_cols_pd = [c for c, t in player_data.dtypes if t in ("double","float","int","bigint")]
for c in num_cols_pd:
    mean_val = player_data.agg(F.avg(c)).first()[0]
    player_data = player_data.withColumn(c,
                                         F.when(F.col(c).isNull(), mean_val).otherwise(F.col(c)))
# Fill missing birth_date with default
if "birth_date" in player_data.columns:
    player_data = player_data.fillna({"birth_date": "1900-01-01"})

# -------- Clean players_df --------
for c, t in players_df.dtypes:
    if t == "string" and c != "height":
        players_df = players_df.fillna({c: "Unknown"})
players_df = players_df.withColumn("height_in", convert_height_udf(F.col("height")))
mean_height_players = players_df.agg(F.avg("height_in")).first()[0]
players_df = players_df.withColumn("height_in",
                                   F.when(F.col("height_in").isNull(), mean_height_players)
                                     .otherwise(F.col("height_in")))
# Impute numeric columns in players_df
num_cols_pl = [c for c, t in players_df.dtypes if t in ("double","float","int","bigint")]
for c in num_cols_pl:
    mean_val = players_df.agg(F.avg(c)).first()[0]
    players_df = players_df.withColumn(c,
                                       F.when(F.col(c).isNull(), mean_val).otherwise(F.col(c)))

# -------- Clean seasons_df --------
# Fill string columns
str_cols_se = [c for c, t in seasons_df.dtypes if t == "string"]
for c in str_cols_se:
    seasons_df = seasons_df.fillna({c: "Unknown"})
# Impute numeric columns with mean
num_cols_se = [c for c, t in seasons_df.dtypes if t in ("double","float","int","bigint")]
for c in num_cols_se:
    mean_val = seasons_df.agg(F.avg(c)).first()[0]
    seasons_df = seasons_df.withColumn(c,
                                       F.when(F.col(c).isNull(), mean_val).otherwise(F.col(c)))
# Standardise positions
if "Pos" in seasons_df.columns:
    seasons_df = seasons_df.withColumn("Pos", F.regexp_replace(F.col("Pos"), "-.*", ""))

# Remove outliers in selected columns (no binning)
outlier_targets = [c for c in ["PTS","TRB","AST","Age"] if c in seasons_df.columns]
for col in outlier_targets:
    stats = seasons_df.agg(F.mean(col).alias("mean"), F.stddev(col).alias("std")).first()
    mu, sigma = stats["mean"], stats["std"]
    if sigma and sigma != 0:
        zscore = (F.col(col) - mu) / sigma
        seasons_df = seasons_df.filter(F.abs(zscore) <= 3)

# Visualise missing values (unchanged from previous version)
def plot_missing(df, title, filename):
    miss = {c: df.filter(F.col(c).isNull()).count() for c in df.columns}
    pd.Series(miss).plot(kind="bar")
    plt.title(title); plt.ylabel("Missing count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{filename}")
    plt.close()

plot_missing(player_data, "Missing values – player_data.csv", "missing_player_data.png")
plot_missing(players_df, "Missing values – Players.csv", "missing_players.png")
miss_seasons = {c: seasons_df.filter(F.col(c).isNull()).count() for c in seasons_df.columns}
miss_sorted  = dict(sorted(miss_seasons.items(), key=lambda x: x[1], reverse=True))
pd.Series(miss_sorted).head(20).plot(kind="bar")
plt.title("Top missing-value columns – Seasons_Stats.csv")
plt.ylabel("Missing count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{figures_dir}/missing_seasons.png")
plt.close()

# Age distribution after cleaning
if "Age" in seasons_df.columns:
    age_sample = seasons_df.select("Age").dropna().sample(fraction=0.1, seed=42).toPandas()
    plt.figure()
    sns.histplot(age_sample["Age"], bins=30)
    plt.title("Age distribution after cleaning")
    plt.xlabel("Age"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/age_after_cleaning.png")
    plt.close()

# Save cleaned datasets
player_data.write.mode("overwrite").csv(clean_player_data_dir, header=True)
players_df.write.mode("overwrite").csv(clean_players_dir, header=True)
seasons_df.write.mode("overwrite").csv(clean_seasons_dir, header=True)

# Summary
print("Cleaning summary:")
print("player_data rows:", player_data.count())
print("Players.csv rows:", players_df.count())
print("Seasons_Stats rows:", seasons_df.count())
print("Cleaning complete.  Cleaned data saved.  Figures saved in", figures_dir)

spark.stop()
