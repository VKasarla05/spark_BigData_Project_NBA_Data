# ------------ 0. IMPORTS -----------
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# ------------ 1. SPARK SESSION -----------
# If you have a Spark cluster master URL, set it here, e.g. "spark://hadoop1:7077"
# Otherwise, use local[*] to use all cores on this VM.
spark = SparkSession.builder \
    .appName("Spark CNN Image Project") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session created with master:", spark.sparkContext.master)

# ------------ 2. LOAD IMAGE PATHS WITH SPARK -----------
# CHANGE THIS TO YOUR REAL PATH ON THE VM
data_path = "/home/your_username/datasets/bone_fracture"  # TODO: update this

df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(data_path)

print("Total images:", df.count())
