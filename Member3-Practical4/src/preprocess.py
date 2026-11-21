
import pandas as pd
from pathlib import Path

# Load raw dataset
df = pd.read_csv("data/raw/digits.csv")

# Select pixel columns
pixel_cols = [
    c for c in df.columns if c.startswith("pixel_")
]

# Normalize pixel values from 0-16 to 0-1
df[pixel_cols] = df[pixel_cols] / 16.0

# Create processed directory
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save processed dataset
df.to_csv("data/processed/digits_processed.csv", index=False)

print("Processed dataset saved:", df.shape)

print(
    "Pixel value range:",
    df[pixel_cols].min().min(),
    "to",
    df[pixel_cols].max().max()
)
