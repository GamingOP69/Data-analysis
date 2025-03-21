import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_synthetic_data(random_state=42):
    """
    Generate a synthetic classification dataset with 1000 samples,
    20 features (10 informative, 5 redundant) and some missing values.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=random_state
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Introduce missing values in a few features (10% missingness)
    rng = np.random.default_rng(random_state)
    for col in feature_names[:5]:
        mask = rng.uniform(0, 1, size=df.shape[0]) < 0.1
        df.loc[mask, col] = np.nan
        
    return df