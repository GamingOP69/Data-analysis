import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustering_analysis(df):
    """
    Perform clustering analysis on the numeric features:
    - Preprocess with imputation and scaling.
    - Use PCA to reduce dimensions for visualization.
    - Apply KMeans clustering, evaluate with silhouette score, and visualize clusters.
    """
    X = df.drop("target", axis=1)

    # Preprocessing: imputation and scaling
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reduce dimensions to 2D for visualization with PCA
    pca_vis = PCA(n_components=2, random_state=42)
    X_pca = pca_vis.fit_transform(X_scaled)

    # Find the optimal number of clusters using silhouette score (try k=2 to 5)
    best_k = 2
    best_score = -1
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette score for k={k}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nOptimal number of clusters based on silhouette score: {best_k}")

    # Run KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Visualize clusters using PCA-reduced data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', legend='full')
    plt.title("KMeans Clustering Visualization (PCA-reduced)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()