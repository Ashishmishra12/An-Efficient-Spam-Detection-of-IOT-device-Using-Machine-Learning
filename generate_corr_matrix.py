import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Ensure directory exists
save_path = "model/static/corr.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Generate sample data for correlation matrix
df = pd.DataFrame(np.random.rand(100, 5), columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"])

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")

# Save the image
plt.savefig(save_path)
plt.close()

# Print debug messages
print(f"Correlation matrix saved at: {save_path}")
print(f"Does file exist? {'Yes' if os.path.exists(save_path) else 'No'}")
