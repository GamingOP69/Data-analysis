import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def supervised_learning_pipeline(df):
    """
    Build a preprocessing and modeling pipeline:
    - Impute missing numeric values, scale data, and reduce dimensions via PCA.
    - Train a RandomForest classifier using GridSearchCV.
    - Print best parameters, classification report, and plot a confusion matrix.
    """
    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Numeric transformation pipeline: imputation and scaling
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    # PCA for dimensionality reduction (retain 95% variance)
    pca = PCA(n_components=0.95, random_state=42)

    # Define classifier (RandomForest)
    clf = RandomForestClassifier(random_state=42)

    # Build the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', pca),
        ('classifier', clf)
    ])

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters from GridSearchCV:")
    print(grid_search.best_params_)

    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    print("\nClassification report on test set:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()