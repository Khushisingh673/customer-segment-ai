import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


#load data

DATA_PATH = "data/Mall_Customers.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

#clean the colms


df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())


#check missing values

print("\nMissing Values:")
print(df.isnull().sum())





print("\nDuplicate Rows:")
print(df.duplicated().sum())




income_column = None
spending_column = None

for column in df.columns:

    lower_name = column.lower()

    if "income" in lower_name:
        income_column = column

    if "spending" in lower_name:
        spending_column = column


if income_column is None or spending_column is None:

    print("\nERROR:")
    print("Could not automatically find Income or Spending Score column.")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        "Income or Spending Score column not found."
    )


print("\nSelected Features:")
print("Income Column:", income_column)
print("Spending Column:", spending_column)




features = [
    income_column,
    spending_column
]

X = df[features].copy()

print("\nSelected Data:")
print(X.head())


#scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed!")

#elbow methid


inertia_values = []

k_values = range(1, 11)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method for Optimal K"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "elbow_method.png"
)

plt.close()

print(
    "\nElbow Method graph saved as elbow_method.png"
)

#SILHOUETTE SCORE

print("\nSilhouette Scores:")

best_k = None
best_score = -1

for k in range(2, 11):

    temp_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = temp_model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K = {k} --> Score = {score:.4f}"
    )

    if score > best_score:

        best_score = score
        best_k = k


print(
    f"\nBest K according to Silhouette Score: {best_k}"
)

print(
    f"Best Silhouette Score: {best_score:.4f}"
)




FINAL_K = 5

print(
    f"\nTraining final K-Means model with K = {FINAL_K}"
)




kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_scaled
)

df["Cluster"] = clusters


print("\nK-Means training completed!")


#original

cluster_centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

print("\nCluster Centers:")

for cluster_id, center in enumerate(cluster_centers):

    print(
        f"Cluster {cluster_id}: "
        f"Income = {center[0]:.2f}, "
        f"Spending = {center[1]:.2f}"
    )



cluster_names = {}

for cluster_id, center in enumerate(cluster_centers):

    income = center[0]
    spending = center[1]

    # High income + high spending
    if income >= 60 and spending >= 60:

        segment_name = "Premium Customers"

    # High income + low spending
    elif income >= 60 and spending < 40:

        segment_name = "Potential Customers"

    # Low income + low spending
    elif income < 45 and spending < 40:

        segment_name = "Budget Customers"

    # Low income + high spending
    elif income < 45 and spending >= 60:

        segment_name = "Active Customers"

    # Medium / balanced customers
    else:

        segment_name = "Regular Customers"

    cluster_names[
        cluster_id
    ] = segment_name


print("\nCluster Name Mapping:")

for cluster_id, name in cluster_names.items():

    print(
        f"Cluster {cluster_id} --> {name}"
    )



df["Segment"] = df["Cluster"].map(
    cluster_names
)




print("\nCustomer Segment Summary:")

segment_summary = (
    df.groupby("Segment")[features]
    .agg(
        [
            "count",
            "mean"
        ]
    )
)

print(
    segment_summary
)




print("\nCustomers Per Segment:")

print(
    df["Segment"].value_counts()
)



OUTPUT_DATA_PATH = (
    "data/segmented_customers.csv"
)

df.to_csv(
    OUTPUT_DATA_PATH,
    index=False
)

print(
    f"\nProcessed dataset saved to {OUTPUT_DATA_PATH}"
)



joblib.dump(
    kmeans,
    "kmeans_model.pkl"
)

print(
    "K-Means model saved as kmeans_model.pkl"
)




joblib.dump(
    scaler,
    "scaler.pkl"
)

print(
    "Scaler saved as scaler.pkl"
)




joblib.dump(
    cluster_names,
    "cluster_names.pkl"
)

print(
    "Cluster names saved as cluster_names.pkl"
)




joblib.dump(
    features,
    "feature_names.pkl"
)

print(
    "Feature names saved as feature_names.pkl"
)




print(
    "\n------------------------------------"
)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "------------------------------------"
)

print(
    "\nGenerated files:"
)

print(
    "1. kmeans_model.pkl"
)

print(
    "2. scaler.pkl"
)

print(
    "3. cluster_names.pkl"
)

print(
    "4. feature_names.pkl"
)

print(
    "5. elbow_method.png"
)

print(
    "6. data/segmented_customers.csv"
)