from sklearn.datasets import load_digits
import pandas as pd
from pathlib import Path

digits = load_digits()

df = pd.DataFrame(
    digits.data,
    columns=[f"pixel_{i}" for i in range(64)]
)

df["target"] = digits.target

Path("data").mkdir(exist_ok=True)

df.to_csv("data/digits.csv", index=False)

print("Dataset saved: data/digits.csv")
print("Shape:", df.shape)
print(df.head())