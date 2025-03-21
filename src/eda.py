import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform exploratory data analysis:
    - Print summary statistics
    - Plot histograms for sample features
    - Plot a correlation heatmap (excluding the target column)
    """
    print("DataFrame shape:", df.shape)
    print("\nDataFrame summary:")
    print(df.describe())

    # Plot distributions for a subset of features
    sample_features = df.columns[:5]
    df[sample_features].hist(bins=20, figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.drop('target', axis=1).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()