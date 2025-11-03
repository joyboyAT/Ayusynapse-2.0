import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AyuSynapseTrainer:
    def __init__(self, processed_data_dir="c:/AyuSynapse/Dataset/processed/"):
        self.processed_data_dir = processed_data_dir
        self.models_dir = "c:/AyuSynapse/models/"
        self.results_dir = "c:/AyuSynapse/results/"
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load processed training and test data"""
        print("Loading processed data...")
        
        try:
            self.X_train = pd.read_csv(os.path.join(self.processed_data_dir, 'X_train.csv'))
            self.X_test = pd.read_csv(os.path.join(self.processed_data_dir, 'X_test.csv'))
            self.y_train = pd.read_csv(os.path.join(self.processed_data_dir, 'y_train.csv'))
            self.y_test = pd.read_csv(os.path.join(self.processed_data_dir, 'y_test.csv'))
            
            # Convert to numpy arrays if needed
            if 'target_combined' in self.y_train.columns:
                self.y_train = self.y_train['target_combined'].values
            if 'target_combined' in self.y_test.columns:
                self.y_test = self.y_test['target_combined'].values
            
            print(f"✓ Data loaded successfully!")
            print(f"  Training set: {self.X_train.shape}")
            print(f"  Test set: {self.X_test.shape}")
            
            # Handle target distribution safely
            try:
                from collections import Counter
                target_counts = Counter(self.y_train)
                print(f"  Target distribution: {dict(target_counts)}")
            except Exception as e:
                print(f"  Target values: {np.unique(self.y_train)}")
                print(f"  Could not calculate distribution: {e}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        
        return True
    
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        print("\nInitializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=42,
                probability=True
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500
            )
        }
        
        print(f"✓ Initialized {len(self.models)} models")
    
    def train_models(self):
        """Train all models and evaluate performance"""
        print("\nTraining models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(self.y_train, y_pred_train)
                test_accuracy = accuracy_score(self.y_test, y_pred_test)
                f1 = f1_score(self.y_test, y_pred_test, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred_test
                }
                
                print(f"  ✓ {name} trained successfully!")
                print(f"    Train Accuracy: {train_accuracy:.4f}")
                print(f"    Test Accuracy: {test_accuracy:.4f}")
                print(f"    F1 Score: {f1:.4f}")
                print(f"    CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
    
    def save_best_model(self):
        """Save the best performing model"""
        if not self.results:
            print("No models trained yet!")
            return
        
        # Find best model based on test accuracy
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.results[best_model_name]['model']
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'ayusynapse_best_model.pkl')
        joblib.dump(best_model, model_path)
        
        # Save model info
        model_info = {
            'model_name': best_model_name,
            'test_accuracy': self.results[best_model_name]['test_accuracy'],
            'f1_score': self.results[best_model_name]['f1_score'],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        info_path = os.path.join(self.models_dir, 'model_info.txt')
        with open(info_path, 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Best model ({best_model_name}) saved to {model_path}")
        print(f"  Test Accuracy: {model_info['test_accuracy']:.4f}")
        
        return best_model_name, best_model
    
    def generate_reports(self):
        """Generate detailed classification reports and visualizations"""
        print("\nGenerating reports...")
        
        # Create results summary
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train_Accuracy': [self.results[name]['train_accuracy'] for name in self.results.keys()],
            'Test_Accuracy': [self.results[name]['test_accuracy'] for name in self.results.keys()],
            'F1_Score': [self.results[name]['f1_score'] for name in self.results.keys()],
            'CV_Mean': [self.results[name]['cv_mean'] for name in self.results.keys()],
            'CV_Std': [self.results[name]['cv_std'] for name in self.results.keys()]
        })
        
        # Save results
        results_path = os.path.join(self.results_dir, 'model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        
        # Print detailed results to console
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        # Generate classification report for best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        best_predictions = self.results[best_model_name]['predictions']
        
        print(f"\n" + "=" * 60)
        print(f"DETAILED REPORT - BEST MODEL: {best_model_name}")
        print("=" * 60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_predictions))
        
        # Confusion matrix (text format)
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, best_predictions)
        unique_labels = np.unique(np.concatenate([self.y_test, best_predictions]))
        
        # Print confusion matrix with labels
        print(f"{'':>15}", end="")
        for label in unique_labels:
            print(f"{label:>10}", end="")
        print()
        
        for i, true_label in enumerate(unique_labels):
            print(f"{true_label:>15}", end="")
            for j in range(len(unique_labels)):
                print(f"{cm[i, j]:>10}", end="")
            print()
        
        # Feature importance for tree-based models
        best_model = self.results[best_model_name]['model']
        if hasattr(best_model, 'feature_importances_'):
            print("\nTop 10 Most Important Features:")
            feature_names = self.X_train.columns
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"  {i+1:2d}. {feature_names[idx][:40]:<40} ({importances[idx]:.4f})")
        
        print(f"\n✓ Detailed results saved to {results_path}")
        
        # Try to create basic visualization if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Model comparison
            plt.subplot(2, 3, 1)
            models = results_df['Model']
            test_acc = results_df['Test_Accuracy']
            plt.bar(range(len(models)), test_acc)
            plt.title('Model Test Accuracy Comparison')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: F1 Scores
            plt.subplot(2, 3, 2)
            plt.bar(range(len(models)), results_df['F1_Score'])
            plt.title('Model F1 Score Comparison')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylabel('F1 Score')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Cross-validation scores
            plt.subplot(2, 3, 3)
            plt.errorbar(range(len(models)), results_df['CV_Mean'], 
                        yerr=results_df['CV_Std'], fmt='o', capsize=5)
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.title('Cross-Validation Scores')
            plt.ylabel('CV Score')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Train vs Test Accuracy
            plt.subplot(2, 3, 4)
            x = np.arange(len(models))
            width = 0.35
            plt.bar(x - width/2, results_df['Train_Accuracy'], width, label='Train', alpha=0.8)
            plt.bar(x + width/2, results_df['Test_Accuracy'], width, label='Test', alpha=0.8)
            plt.title('Train vs Test Accuracy')
            plt.xticks(x, models, rotation=45, ha='right')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Confusion Matrix (simple)
            plt.subplot(2, 3, 5)
            im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(f'Confusion Matrix\n{best_model_name}')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            tick_marks = np.arange(len(unique_labels))
            plt.xticks(tick_marks, unique_labels)
            plt.yticks(tick_marks, unique_labels)
            
            # Plot 6: Model Performance Summary
            plt.subplot(2, 3, 6)
            metrics = ['Test_Accuracy', 'F1_Score', 'CV_Mean']
            best_idx = results_df['Test_Accuracy'].idxmax()
            best_metrics = [results_df.loc[best_idx, metric] for metric in metrics]
            
            plt.bar(metrics, best_metrics, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title(f'Best Model Performance\n{best_model_name}')
            plt.ylabel('Score')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(best_metrics):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = os.path.join(self.results_dir, 'training_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Visualizations saved to {plot_path}")
            
        except ImportError:
            print("✓ Matplotlib not available, skipping visualizations")
            print("  Install matplotlib for plots: pip install matplotlib")
        except Exception as e:
            print(f"✓ Visualization error (continuing anyway): {e}")

    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("=" * 50)
        print("AyuSynapse Model Training Pipeline")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models()
        
        # Save best model
        best_model_name, best_model = self.save_best_model()
        
        # Generate reports
        self.generate_reports()
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Best Model: {best_model_name}")
        print("=" * 50)

def main():
    trainer = AyuSynapseTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
