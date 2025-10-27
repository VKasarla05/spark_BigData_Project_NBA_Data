# Data Discretization

from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer, VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_style("whitegrid")
FIG_OUTPUT = "/home/sat3812/discretization_figures_fast"
pathlib.Path(FIG_OUTPUT).mkdir(parents=True, exist_ok=True)

# Spark init
spark = SparkSession.builder.appName("NBADataDiscretization_Fast").getOrCreate()

# Load transformed data
DATA_PATH = "/home/sat3812/clean_output/transformed_nba_stats"
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df = df.sample(fraction=0.25, seed=42)

numeric_cols = [col for col, dtype in df.dtypes 
                if dtype in ('double', 'float', 'int', 'bigint')]

print(f"Fast discretization on {len(numeric_cols)} numeric columns (sampled data)")
print(f"Dataset: {df.count():,} rows (25% sample)")


### METHOD 1: Equal-Width Binning ###
print("\n" + "="*60)
print("Method 1: Equal-Width Binning")
print("="*60)

# Equal-width bins
for col in numeric_cols[:3]: 
    try:
        # Get min/max for the column
        stats = df.selectExpr(f"min({col}) as min", f"max({col}) as max").first()
        min_val = stats['min']
        max_val = stats['max']

        if min_val is None or max_val is None or min_val == max_val:
            print(f"  Skipping {col} (insufficient range)")
            continue

        # Create 5 equal-width bins
        num_bins = 5
        bin_width = (max_val - min_val) / num_bins

        # Define split points
        splits = [min_val + i * bin_width for i in range(num_bins + 1)]
        splits[0] = float('-inf')  
        splits[-1] = float('inf')  

        # Apply bucketizer
        bucketizer = Bucketizer(
            splits=splits,
            inputCol=col,
            outputCol=f'{col}_equal_width'
        )

        df = bucketizer.transform(df)
        print(f" {col}: binned into {num_bins} equal-width categories")

    except Exception as e:
        print(f" Error binning {col}: {str(e)[:50]}...")


### METHOD 2: Quantile-Based Binning ###  
print("\n" + "="*60)
print("Method 2: Quantile Binning (Equal Frequency)")
print("="*60)

# Quantile binning 
for col in numeric_cols[3:6]: 
    try:
        discretizer = QuantileDiscretizer(
            numBuckets=5,
            inputCol=col,
            outputCol=f'{col}_quantile',
            handleInvalid='keep' 
        )

        disc_model = discretizer.fit(df)
        df = disc_model.transform(df)

        print(f"  {col}: binned into 5 quantiles")

    except Exception as e:
        print(f"  Error with quantile binning for {col}: {str(e)[:50]}...")


### METHOD 3: K-Means Clustering ###
print("\n" + "="*60)
print("Method 3: K-Means Clustering (Automatic Segmentation)")
print("="*60)

cluster_cols = numeric_cols[:5]

print(f"Clustering on features: {cluster_cols}")

# Assemble features
assembler = VectorAssembler(
    inputCols=cluster_cols,
    outputCol='cluster_features',
    handleInvalid='skip'
)

df = assembler.transform(df)

# Trying different values of k to find elbow
k_values = [3, 4, 5, 6, 7]
inertias = []

print("\nTesting different cluster counts")
for k in k_values:
    kmeans = KMeans(
        k=k,
        featuresCol='cluster_features',
        predictionCol='cluster',
        seed=42
    )

    model = kmeans.fit(df)
    cost = model.summary.trainingCost 
    inertias.append(cost)

    print(f"  k={k}: cost={cost:.2f}")

# Plot elbow curve 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, inertias, 'o-', linewidth=2, markersize=10, color='rebeccapurple')
ax.set_xlabel('Number of Clusters (k)', fontsize=11)
ax.set_ylabel('Within-Cluster Sum of Squares', fontsize=11)
ax.set_title('K-Means Elbow Method', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values)

plt.tight_layout()
elbow_path = f'{FIG_OUTPUT}/kmeans_elbow.png'
plt.savefig(elbow_path, dpi=120)
plt.close()

print(f"\n  ✓ Saved elbow plot: {elbow_path}")

# Using k based on elbow 
optimal_k = 4
print(f"\nApplying K-Means with k={optimal_k}...")

kmeans_final = KMeans(
    k=optimal_k,
    featuresCol='cluster_features',
    predictionCol='player_cluster',
    seed=42,
    maxIter=20
)

kmeans_model = kmeans_final.fit(df)
df = kmeans_model.transform(df)

# Check cluster sizes
cluster_counts = df.groupBy('player_cluster').count().orderBy('player_cluster').collect()

print("\nCluster distribution:")
for row in cluster_counts:
    cluster_id = row['player_cluster']
    count = row['count']
    pct = (count / df.count()) * 100
    print(f"  Cluster {cluster_id}: {count:,} players ({pct:.1f}%)")


# Visualize clusters
if len(cluster_cols) >= 2:
    print("\nGenerating cluster visualization...")

    viz_df = df.select(cluster_cols[0], cluster_cols[1], 'player_cluster').toPandas()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plotting
    cluster_colors = ['dodgerblue', 'lightcoral', 'gold', 'mediumorchid']
    for cluster_id in range(optimal_k):
        cluster_data = viz_df[viz_df['player_cluster'] == cluster_id]
        ax.scatter(
            cluster_data[cluster_cols[0]], 
            cluster_data[cluster_cols[1]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=40,
            color=cluster_colors[cluster_id % len(cluster_colors)],
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(cluster_cols[0], fontsize=11)
    ax.set_ylabel(cluster_cols[1], fontsize=11)
    ax.set_title(f'Player Segmentation: K-Means Clustering (k={optimal_k})', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    cluster_viz_path = f'{FIG_OUTPUT}/kmeans_clusters_2d.png'
    plt.savefig(cluster_viz_path, dpi=120)
    plt.close()

    print(f" Saved cluster plot: {cluster_viz_path}")


# Comparision
print("\nGenerating cluster comparison boxplot...")

comparison_col = cluster_cols[0]
compare_df = df.select(comparison_col, 'player_cluster').toPandas()

fig, ax = plt.subplots(figsize=(9, 6))
compare_df.boxplot(column=comparison_col, by='player_cluster', ax=ax)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel(comparison_col, fontsize=11)
ax.set_title(f'{comparison_col} Distribution by Cluster', fontsize=12)
plt.suptitle('') 
plt.tight_layout()
boxplot_path = f'{FIG_OUTPUT}/cluster_feature_comparison.png'
plt.savefig(boxplot_path, dpi=120)
plt.close()

print(f" Saved comparison: {boxplot_path}")


print("\nCleaning up intermediate columns...")
df = df.drop('cluster_features')


# Save discretized dataset
OUTPUT_PATH = "/home/sat3812/discretized_nba_stats"
print(f"\nSaving discretized dataset to: {OUTPUT_PATH}")

df.write.csv(OUTPUT_PATH, header=True, mode='overwrite')

print("\n" + "="*60)
print("Discretization Complete!")
print("="*60)
print(f"Output: {OUTPUT_PATH}")
print(f"Visualizations: {FIG_OUTPUT}/")
print(f"\nCreated:")
print(f"  • {3} equal-width binned features")
print(f"  • {3} quantile-binned features")
print(f"  • {optimal_k} player clusters from K-means")

spark.stop()

