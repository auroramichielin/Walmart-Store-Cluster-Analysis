import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET = "Walmart_Store_sales.csv"
K = 4

# ============================================================
# CARICAMENTO DATI
# ============================================================

print("\n==============================")
print("CARICAMENTO DATASET")
print("==============================")

df = pd.read_csv(DATASET)

store_profile = (
    df.groupby("Store")
      .agg(
          sales=("Weekly_Sales", "mean"),
          temp=("Temperature", "mean"),
          fuel=("Fuel_Price", "mean"),
          cpi=("CPI", "mean"),
          unemp=("Unemployment", "mean")
      )
      .reset_index()
)

print(f"Numero store: {len(store_profile)}")

# ============================================================
# STANDARDIZZAZIONE
# ============================================================

FEATURES = [
    "sales",
    "temp",
    "fuel",
    "cpi",
    "unemp"
]

scaler = StandardScaler()
X = scaler.fit_transform(store_profile[FEATURES])

print("\nStandardizzazione completata.")

# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

print("\n==============================")
print("SILHOUETTE ANALYSIS")
print("==============================")

K_range = range(2, 11)
silhouettes = []

for k in K_range:

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42
    )

    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)

    silhouettes.append(sil)

    print(f"K={k} -> Silhouette={sil:.4f}")

plt.figure(figsize=(8,5))

plt.plot(
    list(K_range),
    silhouettes,
    marker="o",
    linewidth=2
)

plt.axvline(
    K,
    linestyle="--",
    label=f"K scelto = {K}"
)

plt.title("Silhouette Analysis")
plt.xlabel("Numero Cluster")
plt.ylabel("Silhouette Score")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("1_silhouette_analysis.png", dpi=150)
plt.show()

# ============================================================
# K-MEANS
# ============================================================

print("\n==============================")
print("K-MEANS")
print("==============================")

kmeans = KMeans(
    n_clusters=K,
    n_init=20,
    random_state=42
)

km_labels = kmeans.fit_predict(X)

km_silhouette = silhouette_score(
    X,
    km_labels
)

print(f"Silhouette Score: {km_silhouette:.4f}")

cluster_sizes_km = (
    pd.Series(km_labels)
    .value_counts()
    .sort_index()
)

print("\nStore per cluster:")

for cluster, size in cluster_sizes_km.items():
    print(f"Cluster {cluster}: {size}")

# Scatter KMeans

plt.figure(figsize=(8,6))

for c in range(K):

    mask = km_labels == c

    plt.scatter(
        store_profile["sales"][mask],
        store_profile["cpi"][mask],
        s=80,
        label=f"Cluster {c}"
    )

plt.title("K-Means Clustering")
plt.xlabel("Average Sales")
plt.ylabel("Average CPI")
plt.legend()

plt.tight_layout()
plt.savefig("2_kmeans_scatter.png", dpi=150)
plt.show()

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

print("\n==============================")
print("HIERARCHICAL (WARD)")
print("==============================")

hc = AgglomerativeClustering(
    n_clusters=K,
    linkage="ward"
)

hc_labels = hc.fit_predict(X)

hc_silhouette = silhouette_score(
    X,
    hc_labels
)

print(f"Silhouette Score: {hc_silhouette:.4f}")

cluster_sizes_hc = (
    pd.Series(hc_labels)
    .value_counts()
    .sort_index()
)

print("\nStore per cluster:")

for cluster, size in cluster_sizes_hc.items():
    print(f"Cluster {cluster}: {size}")

# Dendrogramma

Z = linkage(
    X,
    method="ward"
)

plt.figure(figsize=(14,6))

dendrogram(
    Z,
    labels=[f"S{s}" for s in store_profile["Store"]],
    leaf_font_size=8
)

plt.title("Dendrogramma Ward")
plt.ylabel("Distanza")

plt.tight_layout()
plt.savefig("3_dendrogramma_ward.png", dpi=150)
plt.show()

# ============================================================
# DBSCAN
# ============================================================

print("\n==============================")
print("DBSCAN")
print("==============================")

db = DBSCAN(
    eps=1.8,
    min_samples=3
)

db_labels = db.fit_predict(X)

n_clusters_db = (
    len(set(db_labels))
    - (1 if -1 in db_labels else 0)
)

n_outliers = list(db_labels).count(-1)

print(f"Cluster trovati: {n_clusters_db}")
print(f"Outlier trovati: {n_outliers}")

cluster_sizes_db = (
    pd.Series(db_labels)
    .value_counts()
    .sort_index()
)

print("\nStore per cluster:")

for cluster, size in cluster_sizes_db.items():

    if cluster == -1:
        print(f"Outlier: {size}")
    else:
        print(f"Cluster {cluster}: {size}")

# Silhouette DBSCAN

if n_clusters_db > 1:

    mask = db_labels != -1

    db_silhouette = silhouette_score(
        X[mask],
        db_labels[mask]
    )

else:

    db_silhouette = 0

print(f"\nSilhouette Score: {db_silhouette:.4f}")

# Scatter DBSCAN

plt.figure(figsize=(8,6))

for lbl in sorted(set(db_labels)):

    mask = db_labels == lbl

    if lbl == -1:

        plt.scatter(
            store_profile["sales"][mask],
            store_profile["cpi"][mask],
            marker="X",
            s=120,
            label="Outlier"
        )

    else:

        plt.scatter(
            store_profile["sales"][mask],
            store_profile["cpi"][mask],
            s=80,
            label=f"Cluster {lbl}"
        )

plt.title("DBSCAN")
plt.xlabel("Average Sales")
plt.ylabel("Average CPI")
plt.legend()

plt.tight_layout()
plt.savefig("4_dbscan_scatter.png", dpi=150)
plt.show()

# ============================================================
# CONFRONTO FINALE
# ============================================================

comparison = pd.DataFrame({
    "Algoritmo": [
        "K-Means",
        "Ward",
        "DBSCAN"
    ],
    "Silhouette": [
        km_silhouette,
        hc_silhouette,
        db_silhouette
    ]
})

print("\n==============================")
print("CONFRONTO FINALE")
print("==============================")
print(comparison.round(4))

plt.figure(figsize=(7,4))

plt.bar(
    comparison["Algoritmo"],
    comparison["Silhouette"]
)

plt.title("Confronto Silhouette")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("5_confronto_algoritmi.png", dpi=150)
plt.show()

# ============================================================
# FILE GENERATI
# ============================================================

print("\n==============================")
print("FILE GENERATI")
print("==============================")

print("1_silhouette_analysis.png")
print("2_kmeans_scatter.png")
print("3_dendrogramma_ward.png")
print("4_dbscan_scatter.png")
print("5_confronto_algoritmi.png")