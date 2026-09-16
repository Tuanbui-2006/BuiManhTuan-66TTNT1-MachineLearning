import pandas as pd

df = pd.read_csv("du_lieu_gia_nha_200_dong.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
