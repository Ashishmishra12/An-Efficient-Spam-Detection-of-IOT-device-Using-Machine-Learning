import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Ensure directory exists
save_path = "model/static/conf_matrix.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Sample confusion matrix data
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Spam", "Spam"], yticklabels=["No Spam", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save the image
plt.savefig(save_path)
plt.close()

# Print debug messages
print(f"Confusion matrix saved at: {save_path}")
print(f"Does file exist? {'Yes' if os.path.exists(save_path) else 'No'}")
