
import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/digits_processed.csv")

# Select pixel columns
pixel_cols = [
    c for c in df.columns if c.startswith("pixel_")
]

# Validate dataset
assert len(df) == 1797, "Unexpected number of rows"
assert len(pixel_cols) == 64, "Expected 64 pixel features"
assert df["target"].between(0, 9).all(), "Target must be 0 to 9"
assert not df.isnull().any().any(), "Missing values found"
assert df[pixel_cols].min().min() >= 0
assert df[pixel_cols].max().max() <= 1

print("Validation successful")
print("Rows:", len(df), "| Features:", len(pixel_cols))
