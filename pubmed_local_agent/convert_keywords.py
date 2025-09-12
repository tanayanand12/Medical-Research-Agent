import pandas as pd

df = pd.read_csv("pubmeds_keywords.csv")

lst_keywords = df["Pubmed keywords"].tolist()
print(lst_keywords)
