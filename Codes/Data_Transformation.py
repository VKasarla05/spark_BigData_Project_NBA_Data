# Data Transformation 
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.functions import vector_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_palette("Set2")
fig_dir = "transformation_figures"
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("NBADataTransformation").getOrCreate()

# Loading Dataset
df = spark.read.csv("/home/sat3812/reduced_nba_stats", header=True, inferSchema=True)

# Identifying numeric and categorical columns
num_cols = [c for c, t in df.dtypes if t in ("double", "float", "int", "bigint")]
cat_cols = [c for c, t in df.dtypes if t == "string" and c != "name_norm"]

print(f"Numeric columns: {len(num_cols)} | Categorical columns: {len(cat_cols)}")

# Numeric Transformation 
assembler = VectorAssembler(inputCols=num_cols, outputCol="features_vec", handleInvalid="keep")
df = assembler.transform(df)

# MinMax Scaling
minmax_scaler = MinMaxScaler(inputCol="features_vec", outputCol="features_minmax")
df = minmax_scaler.fit(df).transform(df)

# Standard Scaling
std_scaler = StandardScaler(inputCol="features_vec", outputCol="features_standard", withMean=True, withStd=True)
df = std_scaler.fit(df).transform(df)

# Convert scaled vectors to arrays
df = df.withColumn("minmax_array", vector_to_array("features_minmax")) \
       .withColumn("standard_array", vector_to_array("features_standard"))

# Categorical Encoding
for c in cat_cols[:3]:
    indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip")
    encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    df = indexer.fit(df).transform(df)
    df = encoder.fit(df).transform(df)

# Visualizations 
if num_cols:
    col_name = num_cols[0]
    sample_pdf = df.select(col_name).dropna().sample(fraction=0.05, seed=42).toPandas()

    # Before scaling
    plt.figure(figsize=(6, 4))
    sns.histplot(sample_pdf[col_name], bins=30, color="#5DADE2")
    plt.title(f"{col_name} - Before Scaling")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{col_name}_before_scaling.png")
    plt.close()

    # After scaling (MinMax)
    pdf_scaled = pd.DataFrame(df.select("minmax_array").limit(1000).toPandas()["minmax_array"].to_list())
    sns.histplot(pdf_scaled.iloc[:, 0], bins=30, color="#E74C3C")
    plt.title(f"{col_name} - After MinMax Scaling")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{col_name}_after_scaling.png")
    plt.close()

# Drop Unsupported Columns Before Saving
drop_cols = [ "features_vec","features_minmax","features_standard","minmax_array","standard_array",] + [c for c in df.columns if c.endswith("_ohe")]
df = df.drop(*[c for c in drop_cols if c in df.columns])

# Save Transformed Dataset
output_path = "clean_output/transformed_nba_stats"
df.write.mode("overwrite").csv(output_path, header=True)

# Summary Output
print("Data Transformation Complete")
print(f"Records: {df.count()} | Columns: {len(df.columns)}")
print(f"Output saved to: {output_path}")
print(f"Visualizations saved in: {fig_dir}")

spark.stop()


