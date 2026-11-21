
from sklearn.datasets import load_digits
import pandas as pd
from pathlib import Path

# Load handwritten digits dataset
digits = load_digits()

# Convert dataset into DataFrame
df = pd.DataFrame(
    digits.data,
    columns=[f"pixel_{i}" for i in range(64)]
)

# Add target column
df["target"] = digits.target

# Create raw data directory
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Save dataset
df.to_csv("data/raw/digits.csv", index=False)

print("Raw dataset created:", df.shape)
