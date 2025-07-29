import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           precision_score, recall_score)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_explore_data(self, data_path=None):
        """Load and explore the dataset"""
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            # Generate synthetic credit card data for demonstration
            print("Generating synthetic credit card fraud dataset...")
            self.df = self._generate_synthetic_data()
        
        print("Dataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Fraud cases: {self.df['Class'].sum()}")
        print(f"Normal cases: {len(self.df) - self.df['Class'].sum()}")
        print(f"Fraud percentage: {self.df['Class'].mean()*100:.2f}%")
        
        return self.df
    
    def _generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic credit card transaction data"""
        np.random.seed(42)
        
        # Generate features similar to credit card data
        features = {}
        
        # Time feature (seconds elapsed between transactions)
        features['Time'] = np.random.exponential(1000, n_samples)
        
        # Amount feature (transaction amount)
        normal_amounts = np.random.lognormal(3, 1.5, int(n_samples * 0.999))
        fraud_amounts = np.random.lognormal(4, 2, int(n_samples * 0.001))
        features['Amount'] = np.concatenate([normal_amounts, fraud_amounts])
        
        # Generate V1-V28 features (PCA transformed features)
        for i in range(1, 29):
            if i <= 14:
                # First half features
                normal_vals = np.random.normal(0, 1, int(n_samples * 0.999))
                fraud_vals = np.random.normal(2, 1.5, int(n_samples * 0.001))
            else:
                # Second half features
                normal_vals = np.random.normal(0, 1, int(n_samples * 0.999))
                fraud_vals = np.random.normal(-1.5, 1.2, int(n_samples * 0.001))
            
            features[f'V{i}'] = np.concatenate([normal_vals, fraud_vals])
        
        # Create target variable
        features['Class'] = np.concatenate([
            np.zeros(int(n_samples * 0.999)),
            np.ones(int(n_samples * 0.001))
        ])
        
        # Shuffle the data
        df = pd.DataFrame(features)
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def visualize_data(self):
        """Create visualizations for exploratory data analysis"""
        plt.figure(figsize=(20, 15))
        
        # Class distribution
        plt.subplot(2, 3, 1)
        self.df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Class Distribution')
        plt.xlabel('Class (0: Normal, 1: Fraud)')
        plt.ylabel('Count')
        
        # Amount distribution by class
        plt.subplot(2, 3, 2)
        self.df[self.df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, label='Normal', color='skyblue')
        self.df[self.df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud', color='salmon')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.title('Amount Distribution by Class')
        plt.legend()
        plt.yscale('log')
        
        # Time distribution by class
        plt.subplot(2, 3, 3)
        self.df[self.df['Class'] == 0]['Time'].hist(bins=50, alpha=0.7, label='Normal', color='skyblue')
        self.df[self.df['Class'] == 1]['Time'].hist(bins=50, alpha=0.7, label='Fraud', color='salmon')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Time Distribution by Class')
        plt.legend()
        
        # Correlation heatmap for selected features
        plt.subplot(2, 3, 4)
        corr_features = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'Class']
        correlation_matrix = self.df[corr_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (Selected Features)')
        
        # Box plot for amount by class
        plt.subplot(2, 3, 5)
        sns.boxplot(data=self.df, x='Class', y='Amount')
        plt.title('Amount Distribution by Class (Box Plot)')
        plt.yscale('log')
        
        # Feature importance preview (using a quick Random Forest)
        plt.subplot(2, 3, 6)
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 10 Feature Importances (Random Forest)')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self, sampling_strategy='smote'):
        """Preprocess the data with scaling and resampling"""
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance
        if sampling_strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        elif sampling_strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_scaled, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
        self.X_train = X_train_resampled
        self.X_test = X_test_scaled
        self.y_train = y_train_resampled
        self.y_test = y_test
        
        print(f"Training set shape after resampling: {self.X_train.shape}")
        print(f"Training set class distribution: {np.bincount(self.y_train)}")
        
    def train_models(self):
        """Train multiple models for fraud detection"""
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        print("Training models...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
        print("All models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = []
        
        plt.figure(figsize=(15, 10))
        
        # ROC Curves
        plt.subplot(2, 2, 1)
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Precision-Recall Curves
        plt.subplot(2, 2, 2)
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'{name}')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        # Model Performance Comparison
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc
            })
        
        results_df = pd.DataFrame(results)
        
        # Performance metrics bar plot
        plt.subplot(2, 2, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        x = np.arange(len(results_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, results_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 2, results_df['Model'], rotation=45)
        plt.legend()
        
        # Confusion matrix for best model
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        y_pred_best = best_model.predict(self.X_test)
        
        plt.subplot(2, 2, 4)
        cm = confusion_matrix(self.y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results = results_df
        print("\nModel Performance Summary:")
        print(results_df.round(4))
        
        return results_df
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the specified model"""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[f'{model_name}_Tuned'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\n" + "="*50)
        print("CREDIT CARD FRAUD DETECTION REPORT")
        print("="*50)
        
        print(f"\nDataset Summary:")
        print(f"- Total transactions: {len(self.df):,}")
        print(f"- Fraudulent transactions: {self.df['Class'].sum():,}")
        print(f"- Fraud rate: {self.df['Class'].mean()*100:.2f}%")
        
        print(f"\nModel Performance Summary:")
        print(self.results.to_string(index=False))
        
        best_model_name = self.results.loc[self.results['F1-Score'].idxmax(), 'Model']
        print(f"\nBest performing model: {best_model_name}")
        print(f"F1-Score: {self.results.loc[self.results['F1-Score'].idxmax(), 'F1-Score']:.4f}")
        
        return self.results

def main():
    """Main execution function"""
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline()
    
    # Load and explore data
    df = pipeline.load_and_explore_data()
    
    # Create visualizations
    pipeline.visualize_data()
    
    # Preprocess data
    pipeline.preprocess_data(sampling_strategy='smote')
    
    # Train models
    pipeline.train_models()
    
    # Evaluate models
    results = pipeline.evaluate_models()
    
    # Hyperparameter tuning for best models
    pipeline.hyperparameter_tuning('Random Forest')
    pipeline.hyperparameter_tuning('XGBoost')
    
    # Re-evaluate with tuned models
    tuned_results = pipeline.evaluate_models()
    
    # Generate final report
    final_report = pipeline.generate_report()
    
    return pipeline, final_report

if __name__ == "__main__":
    pipeline, report = main()
