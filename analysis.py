# This script performs data analysis and visualization on the Iris dataset.
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset

# Load the Iris dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Load the dataset into a pandas DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset to inspect the data
print("First few rows of the dataset:")
print(df.head())

# Check the structure of the dataset (data types, missing values, etc.)
print("\nDataset info:")
print(df.info())

# Task 1: Clean the dataset (Handling missing values)
# If there are any missing values, we drop them. Otherwise, this step can be modified based on the dataset.
df = df.dropna()  # Drops rows with missing values
print("\nDataset info after cleaning (dropping missing values):")
print(df.info())

# Task 2: Basic Data Analysis

# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Perform groupings by species and compute the mean of each numerical column for each group
print("\nMean of numerical columns grouped by species:")
grouped = df.groupby("species").mean()
print(grouped)

# Task 3: Data Visualization

# 1. Line Chart showing trends over time (Index here as placeholder for time)
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.title("Sepal Length over Time (Index as Placeholder)")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()

# 2. Bar Chart: Average Sepal Length per Species
plt.figure(figsize=(10, 6))
df.groupby("species")["sepal_length"].mean().plot(kind="bar", color='skyblue', edgecolor='black')
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.show()

# 3. Histogram of Sepal Length Distribution
plt.figure(figsize=(10, 6))
df["sepal_length"].plot(kind="hist", bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.show()

# 4. Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(10, 6))
plt.scatter(df["sepal_length"], df["petal_length"], color="orange")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# Additional findings or observations from the analysis
print("\nObservations:")
print("1. The average sepal length varies across species.")
print("2. There seems to be a linear relationship between sepal length and petal length.")
print("3. Sepal length appears to have a roughly normal distribution in the dataset.")
