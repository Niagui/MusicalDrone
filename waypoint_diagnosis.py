import pandas as pd

df = pd.read_csv("trajectories.csv")

pd.set_option('display.max_rows', None)
print(df[df["0"]==0][:50])