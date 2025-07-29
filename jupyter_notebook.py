# Credit Card Fraud Detection - Interactive Analysis

"""
Jupyter Notebook version of the fraud detection pipeline
This notebook provides an interactive way to explore the fraud detection process
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# # Credit Card Fraud Detection Analysis
# 
# This notebook demonstrates a comprehensive approach to credit card fraud detection using machine learning.
# 
# ## Table of Contents
# 1. [Data Loading and Exploration](#data-loading)
# 2. [Exploratory Data Analysis](#eda)
# 3. [Data Preprocessing](#preprocessing)
# 4. [Model Training](#training)
# 5. [Model Evaluation](#evaluation)
# 6. [Results and Conclusions](#conclusions)

# %% [markdown]
# ## 1. Data Loading and Exploration {#data-loading}

# %%
def generate_synthetic_fraud_data(n_samples=10000):
    """Generate synthetic credit card fraud data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    features = {}
    
    # Time feature
    features['Time'] = np.random.exponential(1000, n_samples)
    
    # Amount feature with different distributions for fraud/normal
    normal_amounts = np.random.lognormal(3, 1.5, int(n_samples * 0.999))
    fraud_amounts = np.random.lognormal(4, 2, int(n_samples * 0.001))
    features['Amount'] = np.concatenate([normal_amounts, fraud_amounts])
    
    # V1-V28 features (simulating PCA components)
    for i in range(1, 29):
        if i <= 14:
            normal_vals = np.random.normal(0, 1, int(n_samples * 0.999))
            fraud_vals = np.random.normal(2, 1.5, int(n_samples * 0.001))
        else:
            normal_vals = np.random.normal(0, 1, int(n_samples * 0.999))
            fraud_vals = np.random.normal(-1.5, 1.2, int(n_samples * 0.001))
        
        features[f'V{i}'] = np.concatenate([normal_vals, fraud_vals])
    
    # Target variable
    features['Class'] = np.concatenate([
        np.zeros(int(n_samples * 0.999)),
        np.ones(int(n_samples * 0.001))
    ])
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(features)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Load data
print("Loading credit card fraud dataset...")
df = generate_synthetic_fraud_data(50000)

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {len(df) - df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean()*100:.2f}%")

# Display first few rows
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis {#eda}

# %%
# Dataset statistics
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# %%
# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Class distribution
axes[0, 0].bar(['Normal', 'Fraud'], df['Class'].value_counts(), 
               color=['skyblue', 'salmon'])
axes[0, 0].set_title('Class Distribution')
axes[0, 0].set_ylabel('Count')

# Amount distribution by class
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, label='Normal', 
                                   color='skyblue', ax=axes[0, 1])
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud', 
                                   color='salmon', ax=axes[0, 1])
axes[0, 1].set