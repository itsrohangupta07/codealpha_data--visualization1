# codealpha_data--visualization1
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 50)
print("TASK 3: DATA VISUALIZATION")
print("=" * 50)

# Load sample dataset (Titanic dataset) for Data Visualization
df_viz = sns.load_dataset("titanic")
print(
    f"Loaded Titanic Dataset for Visualization. Shape: {df_viz.shape[0]} rows, {df_viz.shape[1]} columns\n"
)

# 1. Bar Chart / Count Plot: Survival by Class and Gender
plt.figure(figsize=(9, 5))
sns.countplot(data=df_viz, x="class", hue="survived", palette="Set2")
plt.title(
    "Survival Count by Passenger Class and Gender",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Passenger Count", fontsize=12)
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

# 2. Histogram / Distribution Plot: Age Distribution by Survival
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df_viz,
    x="age",
    hue="survived",
    multiple="stack",
    bins=30,
    palette="viridis",
)
plt.title(
    "Age Distribution Stacked by Survival Status",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Passenger Count", fontsize=12)
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap for Numerical Variables
plt.figure(figsize=(8, 6))
numeric_df = df_viz.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
plt.title(
    "Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

print("\n--- Task 3 Executed Successfully! ---")
