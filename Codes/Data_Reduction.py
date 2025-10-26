# Data Reduction
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_palette("cool")
fig_dir = "reduction_figures"
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("NBADataReduction").getOrCreate()

# Load integrated dataset
df = spark.read.csv("/home/sat3812/clean_output/integrated_nba_stats", header=True, inferSchema=True)


# Select numeric columns and impute remaining nulls with mean
num_cols = [c for c, t in df.dtypes if t in ("double", "float", "int", "bigint")]

for c in num_cols:
    mean_val = df.agg(F.avg(F.col(c))).first()[0]
    df = df.withColumn(c, F.when(F.col(c).isNull(), mean_val).otherwise(F.col(c)))

# Drop columns with all nulls or constant values
for c in num_cols:
    if df.select(c).distinct().count() <= 1:
        df = df.drop(c)
num_cols = [c for c in num_cols if c in df.columns]

# Correlation-based feature pruning
if num_cols:
    sample_pdf = df.select(num_cols).sample(fraction=0.2, seed=42).toPandas()
    corr = sample_pdf.corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i,j] > 0.85:
                to_drop.add(corr.columns[j])
    df = df.drop(*to_drop)
    num_cols = [c for c in num_cols if c not in to_drop]

    # Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap – Post-Integration Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/correlation_heatmap.png")
    plt.close()

# Principal Component Analysis (PCA)
if num_cols:
    assembler = VectorAssembler(inputCols=num_cols, outputCol="features_vec", handleInvalid="keep")
    assembled = assembler.transform(df)
    pca = PCA(k=len(num_cols), inputCol="features_vec", outputCol="pca_features")
    model = pca.fit(assembled)
    explained = model.explainedVariance.toArray()

    plt.figure()
    plt.plot(range(1, len(num_cols)+1), explained.cumsum(), marker='o', color="steelblue")
    plt.title("Cumulative Variance Explained (PCA)")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/pca_variance.png")
    plt.close()

    # Retain minimum #components for ≥95% variance
    num_keep = next((i+1 for i, v in enumerate(explained.cumsum()) if v >= 0.95), len(num_cols))

    # Apply PCA transform
    pca_model = PCA(k=num_keep, inputCol="features_vec", outputCol="pca_features").fit(assembled)
    reduced = pca_model.transform(assembled).withColumn("pca_array", vector_to_array("pca_features"))
    for i in range(num_keep):
        reduced = reduced.withColumn(f"PC{i+1}", F.col("pca_array")[i])

    reduced = reduced.drop("features_vec", "pca_features", "pca_array")
else:
    print("No numeric columns available for reduction.")
    reduced = df

# Save results
reduced.write.mode("overwrite").csv("clean_output/reduced_nba_stats", header=True)
# Summary 
print(" Data Reduction Complete")
print(f"Numeric features analyzed: {len(num_cols)}")
print(f"Highly correlated columns dropped: {len(to_drop) if num_cols else 0}")
print(f"PCA components retained (≥95% variance): {num_keep if num_cols else 0}")
print("Visualizations saved in:", fig_dir)

spark.stop()

