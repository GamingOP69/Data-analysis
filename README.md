# Data Analysis

This repository contains an advanced, modular Python project for data analysis. It includes synthetic data generation, exploratory data analysis (EDA), a supervised learning pipeline with hyperparameter tuning, and unsupervised clustering analysis.

## Repository Structure

```
Data_analysis/ 
├── README.md
├── requirements.txt
├── setup.py # (optional)
└── src/ 
├── init.py 
├── data_generation.py 
├── eda.py 
├── supervised_learning.py 
├── clustering.py 
└── main.py
```

- **src/**: Contains all the Python modules.
  - `data_generation.py`: Generates synthetic data with missing values.
  - `eda.py`: Performs exploratory data analysis including summary statistics, histograms, and correlation heatmap.
  - `supervised_learning.py`: Constructs a machine learning pipeline with preprocessing, PCA, and RandomForest classifier using GridSearchCV.
  - `clustering.py`: Performs unsupervised clustering using KMeans and visualizes clusters with PCA.
  - `main.py`: The main entry point to run the complete pipeline.
- **requirements.txt**: Lists all Python packages required.
- **README.md**: This file.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/GamingOP69/Data-analysis.git
   cd Data-analysis
   ```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
mac: source venv/bin/activate    
On Windows: venv\Scripts\activate
```

3. Install the required packages

```python
pip install -r requirements.txt
```


Usage

Run the main script to execute the full data analysis pipeline:

python src/main.py

The script will:

Generate synthetic data.

Perform exploratory data analysis and display plots.

Run a supervised learning pipeline (with hyperparameter tuning) and output performance metrics.

Execute a clustering analysis and visualize the clusters.




License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Python libraries: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, and XGBoost.



### (Optional) File: `setup.py`

If you plan to package this project for distribution, you might add a `setup.py` file. Here’s a simple example:

```python
from setuptools import setup, find_packages

setup(
    name='Data_analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'xgboost'
    ],
    entry_points={
        'console_scripts': [
            'Data_analysis=src.main:main',
        ],
    },
    author='Gamingop',
    description='A modular advanced data analysis project with EDA, supervised and unsupervised learning pipelines.',
)
```


