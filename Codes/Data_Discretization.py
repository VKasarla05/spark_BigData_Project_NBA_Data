# Data Discretization 
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer, VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

# Visualization setup
sns.set_style("whitegrid")
fig_dir = "discretization_figures_fast"
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("NBADataDiscretization_Fast").getOrCreate()

# Load Transformed Dataset (sample only 25% for testing)
df = spark.read.csv("/home/sat3812/transformed_nba_stats", header=True, inferSchema=True)
df = df.sample(fraction=0.25, seed=42) 

num_cols = [c for c, t in df.dtypes if t in ("double", "float", "int", "bigint")]
print(f"Fast discretization on {len(num_cols)} numeric columns (sampled data)")

# Equal-Width Binning
for col in num_cols[:3]:
    stats = df.selectExpr(f"min({col}) as min", f"max({col}) as max").first()
    min_val, max_val = stats["min"], stats["max"]
    if min_val is None or max_val is None or min_val == max_val:
        continue
    splits = [min_val + i*(max_val - min_val)/5 for i in range(6)]
    bucketizer = Bucketizer(splits=splits, inputCol=col, outputCol=f"{col}_width_bin")
    df = bucketizer.transform(df)

# Equal-Frequency (Quantile) Binning 
for col in num_cols[:3]:
    discretizer = QuantileDiscretizer(
        numBuckets=4, inputCol=col, outputCol=f"{col}_freq_bin", handleInvalid="skip"
    )
    df = discretizer.fit(df).transform(df)


# KMeans-Based Discretization 
target_col = next((c for c in num_cols if "Age" in c or "PER" in c), num_cols[0])
assembler = VectorAssembler(inputCols=[target_col], outputCol="features")
df_km = assembler.transform(df.dropna(subset=[target_col]))

kmeans = KMeans(k=4, seed=42, featuresCol="features")
model = kmeans.fit(df_km)
df_km = model.transform(df_km)

df = df.join(
    df_km.select(target_col, "prediction").withColumnRenamed("prediction", f"{target_col}_kmeans_bin"),
    on=target_col, how="left"
)

# Visualizations
pdf = df.groupBy(f"{num_cols[0]}_width_bin").count().orderBy(f"{num_cols[0]}_width_bin").toPandas()
plt.figure(figsize=(6,4))
sns.barplot(x=f"{num_cols[0]}_width_bin", y="count", data=pdf, color="#AED6F1", edgecolor="gray")
plt.title(f"{num_cols[0]} – Equal-Width Binning (Sample)")
plt.tight_layout()
plt.savefig(f"{fig_dir}/{num_cols[0]}_width_fast.png")
plt.close()

if f"{target_col}_kmeans_bin" in df.columns:
    pdf_km = df.select(target_col, f"{target_col}_kmeans_bin").dropna().sample(fraction=0.05, seed=42).toPandas()
    plt.figure(figsize=(7,5))
    sns.violinplot(
        data=pdf_km, x=f"{target_col}_kmeans_bin", y=target_col,
        inner="quartile", hue=f"{target_col}_kmeans_bin", palette="pastel", legend=False
    )
    plt.title(f"{target_col} – KMeans Discretization (Sample)")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{target_col}_kmeans_fast.png")
    plt.close()

# Save Discretized Sample Output
df = df.drop("features") if "features" in df.columns else df
output_path = "clean_output/discretized_nba_stats_fast"
df.write.mode("overwrite").csv(output_path, header=True)

# Summary
print("Fast Discretization Complete")
print(f"Target Column: {target_col}")
print(f"Output Path: {output_path}")
print(f"Figures saved to: {fig_dir}")

spark.stop()

