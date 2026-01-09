import pandas as pd

df = pd.read_csv("data/spam.csv")

assert df.columns.tolist() == ["label", "message"]
assert df["label"].isin([0,1]).all()
assert df.isnull().sum().sum() == 0
