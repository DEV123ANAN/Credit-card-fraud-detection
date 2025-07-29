# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms, evaluation metrics, and visualization techniques.

## 🚀 Overview

This project implements a complete fraud detection pipeline that includes:
- Data exploration and visualization
- Multiple machine learning models
- Class imbalance handling
- Comprehensive model evaluation
- Hyperparameter tuning
- Performance visualization

## 📊 Features

- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM
- **Class Imbalance Handling**: SMOTE oversampling and random undersampling
- **Comprehensive Evaluation**: ROC curves, Precision-Recall curves, confusion matrices
- **Visualization**: EDA plots, model comparison charts, performance metrics
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Synthetic Data Generation**: Creates realistic fraud detection dataset for demonstration

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create a virtual environment:
```bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py          # Main script with complete pipeline
├── requirements.txt            # Required Python packages
├── README.md                  # Project documentation
├── data/                      # Data directory (optional)
│   └── creditcard.csv         # Your dataset (if available)
├── results/                   # Generated results and plots
│   ├── fraud_detection_eda.png
│   └── model_evaluation.png
└── notebooks/                 # Jupyter notebooks (optional)
    └── fraud_analysis.ipynb
```

## 🚦 Usage

### Basic Usage

Run the complete fraud detection pipeline:

```bash
python fraud_detection.py
```

### Using Your Own Dataset

If you have your own credit card dataset, place it in the `data/` directory and modify the data loading section:

```python
# In fraud_detection.py, modify the load_and_explore_data method
pipeline = FraudDetectionPipeline()
df = pipeline.load_and_explore_data('data/your_dataset.csv')
```

### Custom Configuration

You can customize various aspects of the pipeline:

```python
# Different sampling strategies
pipeline.preprocess_data(sampling_strategy='smote')  # or 'undersample' or 'none'

# Specific model training
pipeline.train_models()

# Hyperparameter tuning for specific models
pipeline.hyperparameter_tuning('Random Forest')
pipeline.hyperparameter_tuning('XGBoost')
```

## 📈 Models Implemented

1. **Logistic Regression** - Fast, interpretable baseline model
2. **Random Forest** - Ensemble method with feature importance
3. **Gradient Boosting** - Sequential ensemble learning
4. **XGBoost** - Optimized gradient boosting framework
5. **LightGBM** - Fast gradient boosting framework
6. **Support Vector Machine** - Kernel-based classification

## 📊 Evaluation Metrics

The project evaluates models using multiple metrics:

- **Accuracy** - Overall correctness
- **Precision** - True positive rate among predicted positives
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Area under the ROC curve
- **Confusion Matrix** - Detailed classification results

## 🔍 Key Features

### Data Preprocessing
- Standard scaling for numerical features
- SMOTE oversampling for handling class imbalance
- Train-test split with stratification

### Visualization
- Class distribution analysis
- Feature correlation heatmaps
- Amount and time distributions by class
- ROC and Precision-Recall curves
- Model performance comparison charts

### Model Selection
- Cross-validation for robust evaluation
- Grid search for hyperparameter optimization
- Performance comparison across multiple metrics

## 📋 Requirements

- Python 3.7+
- pandas for data manipulation
- scikit-learn for machine learning models
- imbalanced-learn for handling class imbalance
- XGBoost and LightGBM for advanced boosting
- matplotlib and seaborn for visualization

## 🎯 Results

The project generates several outputs:

1. **EDA Visualizations** (`fraud_detection_eda.png`)
   - Data distribution analysis
   - Feature importance plots
   - Correlation matrices

2. **Model Evaluation Charts** (`model_evaluation.png`)
   - ROC curves comparison
   - Precision-Recall curves
   - Performance metrics bar chart
   - Confusion matrix for best model

3. **Performance Summary**
   - Detailed metrics table
   - Best model identification
   - Hyperparameter tuning results

## 🔧 Customization

### Adding New Models

```python
# Add your custom model to the models dictionary in train_models()
models = {
    'Your Model': YourModelClass(parameters),
    # ... existing models
}
```

### Custom Features

```python
# Modify the feature engineering in preprocess_data()
def custom_feature_engineering(self, df):
    # Add your custom features
    df['custom_feature'] = df['Amount'] / df['Time']
    return df
```

## 📚 Dataset Information

The project can work with:
- **Real Credit Card Dataset**: If you have access to a real credit card fraud dataset
- **Synthetic Dataset**: The script generates a realistic synthetic dataset for demonstration
- **Kaggle Credit Card Dataset**: Compatible with the popular Kaggle credit card fraud dataset

Expected dataset format:
- Features: V1, V2, ..., V28 (PCA-transformed features)
- Time: Transaction timestamp
- Amount: Transaction amount
- Class: Target variable (0 = Normal, 1 = Fraud)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Credit card fraud detection research community
- Scikit-learn and XGBoost developers
- Open-source machine learning community

## 📞 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

---

**Note**: This project is for educational and research purposes. Always ensure compliance with data privacy regulations when working with real financial data.
