import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Mall_Customers.csv")

print(df.head())

# Income distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Annual Income (k$)"],
    bins=20
)

plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.title("Annual Income Distribution")

plt.show()