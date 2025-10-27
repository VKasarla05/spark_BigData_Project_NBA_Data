# Data Transformation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.functions import vector_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_palette("Set2")
OUTPUT_FIGURES_DIR = "/home/sat3812/transformation_figures"
pathlib.Path(OUTPUT_FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# Initialize Spark 
spark = SparkSession.builder \
    .appName("NBADataTransformation") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load the integrated dataset from previous step
DATA_PATH = "/home/sat3812/clean_output/reduced_nba_stats"
print(f"Loading data from: {DATA_PATH}")

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
initial_count = df.count()
initial_cols = len(df.columns)
print(f"Loaded {initial_count:,} rows with {initial_cols} columns")


# Identify column types for transformation
numeric_columns = [col_name for col_name, dtype in df.dtypes 
                   if dtype in ('double', 'float', 'int', 'bigint')]

categorical_columns = [col_name for col_name, dtype in df.dtypes 
                      if dtype == 'string' and col_name != 'name_norm']

print(f"\nColumn breakdown:")
print(f"  Numeric: {len(numeric_columns)}")
print(f"  Categorical: {len(categorical_columns)}")


### PART 1: Numeric Transformations ###
print("\n" + "="*60)
print("PART 1: Transforming Numeric Features")
print("="*60)

# MinMax Scaling (0-1 range)
print("\nApplying MinMax scaling...")

if len(numeric_columns) > 0:
    assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol='numeric_features',
        handleInvalid='skip'  # Skip rows with invalid values
    )

    df = assembler.transform(df)

    # Apply MinMax scaler
    minmax_scaler = MinMaxScaler(
        inputCol='numeric_features',
        outputCol='minmax_scaled'
    )

    minmax_model = minmax_scaler.fit(df)
    df = minmax_model.transform(df)

    # Convert back to individual columns for easier use
    df = df.withColumn('minmax_array', vector_to_array('minmax_scaled'))

    for idx, col_name in enumerate(numeric_columns):
        new_col = f'{col_name}_minmax'
        df = df.withColumn(new_col, F.col('minmax_array')[idx])

    print(f"  Created {len(numeric_columns)} minmax-scaled features")


# Standard Scaling (mean=0, std=1) 
print("\nApplying Standard (Z-score) scaling...")

if len(numeric_columns) > 0:
    standard_scaler = StandardScaler(
        inputCol='numeric_features',
        outputCol='standard_scaled',
        withMean=True,  # Center the data
        withStd=True    # Scale to unit variance
    )

    std_model = standard_scaler.fit(df)
    df = std_model.transform(df)

    # Convert back to columns
    df = df.withColumn('standard_array', vector_to_array('standard_scaled'))

    for idx, col_name in enumerate(numeric_columns):
        new_col = f'{col_name}_standard'
        df = df.withColumn(new_col, F.col('standard_array')[idx])

    print(f"Created {len(numeric_columns)} standard-scaled features")


# Visualize scaling effects on a sample feature
if len(numeric_columns) > 0:
    sample_feature = numeric_columns[0]  
    print(f"\nGenerating scaling comparison for '{sample_feature}'...")

    # Get sample data for visualization
    sample_df = df.select(
        sample_feature,
        f'{sample_feature}_minmax',
        f'{sample_feature}_standard'
    ).sample(fraction=0.1, seed=42).toPandas()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original distribution
    axes[0].hist(sample_df[sample_feature].dropna(), bins=30, 
                 color='darkblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Original: {sample_feature}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # MinMax scaled 
    axes[1].hist(sample_df[f'{sample_feature}_minmax'].dropna(), bins=30,
                 color='tomato', edgecolor='black', alpha=0.7)
    axes[1].set_title(f'MinMax Scaled')
    axes[1].set_xlabel('Value')

    # Standard scaled
    axes[2].hist(sample_df[f'{sample_feature}_standard'].dropna(), bins=30,
                 color='forestgreen', edgecolor='black', alpha=0.7)
    axes[2].set_title(f'Standard Scaled')
    axes[2].set_xlabel('Value')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FIGURES_DIR}/scaling_comparison.png', dpi=120)
    plt.close()
    print(f" Saved: {OUTPUT_FIGURES_DIR}/scaling_comparison.png")


### PART 2: Categorical Encodings ###
print("\n" + "="*60)
print("PART 2: Encoding Categorical Features")
print("="*60)

# String Indexing - convert categories to numeric indices
print("\nApplying StringIndexer to categorical features...")

for cat_col in categorical_columns[:3]:  
    try:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f'{cat_col}_index',
            handleInvalid='keep' 
        )

        indexer_model = indexer.fit(df)
        df = indexer_model.transform(df)

        unique_values = indexer_model.labels
        print(f" {cat_col}: {len(unique_values)} unique values")

    except Exception as e:
        print(f"  Error encoding {cat_col}: {e}")


# One-Hot Encoding 
print("\nApplying One-Hot Encoding")

indexed_cols = [f'{col}_index' for col in categorical_columns[:3]]

for indexed_col in indexed_cols:
    if indexed_col in df.columns:
        try:
            encoder = OneHotEncoder(
                inputCol=indexed_col,
                outputCol=f'{indexed_col}_onehot'
            )

            encoder_model = encoder.fit(df)
            df = encoder_model.transform(df)

            print(f"  {indexed_col} -> one-hot encoded")

        except Exception as e:
            print(f" Error one-hot encoding {indexed_col}: {e}")


# Clean up intermediate columns to reduce clutter
print("\nCleaning up intermediate processing columns")
cols_to_drop = ['numeric_features', 'minmax_scaled', 'minmax_array', 
                'standard_scaled', 'standard_array']

existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
if existing_cols_to_drop:
    df = df.drop(*existing_cols_to_drop)
    print(f"  Dropped {len(existing_cols_to_drop)} intermediate columns")


# Final summary
final_count = df.count()
final_cols = len(df.columns)

print("\n" + "="*60)
print("Transformation Summary")
print("="*60)
print(f"Rows: {initial_count:,} -> {final_count:,}")
print(f"Columns: {initial_cols} -> {final_cols}")
print(f"New features created: {final_cols - initial_cols}")


# Save transformed dataset
OUTPUT_PATH = "/home/sat3812/transformed_nba_stats"
print(f"\nSaving transformed data to: {OUTPUT_PATH}")

df.write.csv(OUTPUT_PATH, header=True, mode='overwrite')

print("\n Data Transformation complete!")
print(f"Output directory: {OUTPUT_PATH}")
print(f"Visualizations: {OUTPUT_FIGURES_DIR}/")

spark.stop()
