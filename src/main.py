from data_generation import generate_synthetic_data
from eda import perform_eda
from supervised_learning import supervised_learning_pipeline
from clustering import clustering_analysis

def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Run Exploratory Data Analysis
    perform_eda(df)
    
    # Execute supervised learning pipeline with hyperparameter tuning
    supervised_learning_pipeline(df)
    
    # Perform unsupervised clustering analysis
    clustering_analysis(df)

if __name__ == '__main__':
    main()