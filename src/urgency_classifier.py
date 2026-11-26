"""
Urgency Classifier Module for Stanford MOOC Forum Posts
Implements multiple ML models to predict urgency levels (1-7 scale)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class UrgencyClassifier:
    def __init__(self, data_path='processed_data.csv', classification_type='binary'):
        """Initialize the urgency classifier
        
        Args:
            data_path: Path to the preprocessed data CSV file
            classification_type: Type of classification ('binary', 'ternary', 'multiclass')
                - binary: low (1-3) vs high (4-7) urgency
                - ternary: low (1-2), mid (3-5), high (6-7) urgency  
                - multiclass: original 8-class classification (0-7)
        """
        self.data_path = data_path
        self.classification_type = classification_type
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load the preprocessed data"""
        print("Loading preprocessed data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        
        # Remove rows with missing text or urgency
        self.data = self.data.dropna(subset=['text', 'urgency'])
        print(f"Data shape after removing missing values: {self.data.shape}")
        
        return self.data
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("Preparing features...")
        
        # Text features
        self.X_text = self.data['cleaned_text'].fillna('')
        
        # Numerical features
        numerical_features = [
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'sentiment_polarity', 'sentiment_subjectivity', 'question_marks',
            'exclamation_marks', 'uppercase_ratio'
        ]
        
        # Handle missing numerical features
        for feature in numerical_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].fillna(0)
        
        self.X_numerical = self.data[numerical_features].fillna(0)
        
        # Target variable - convert to discrete classes
        self.y = self.data['urgency'].values
        
        # Convert urgency to discrete classes (1-7)
        self.y_discrete = np.round(self.y).astype(int)
        
        # Group urgency levels based on classification type
        if self.classification_type == 'binary':
            # Binary: low (1-3) vs high (4-7)
            self.y_grouped = np.where(self.y_discrete <= 3, 0, 1)  # 0=low, 1=high
            class_names = ['Low', 'High']
        elif self.classification_type == 'ternary':
            # Ternary: low (1-2), mid (3-5), high (6-7)
            self.y_grouped = np.where(self.y_discrete <= 2, 0,
                                    np.where(self.y_discrete <= 5, 1, 2))  # 0=low, 1=mid, 2=high
            class_names = ['Low', 'Mid', 'High']
        else:
            # Multiclass: original 8-class (0-7)
            self.y_grouped = self.y_discrete
            class_names = [f'Class_{i}' for i in range(8)]
        
        print(f"Text features shape: {self.X_text.shape}")
        print(f"Numerical features shape: {self.X_numerical.shape}")
        print(f"Classification type: {self.classification_type}")
        print(f"Original target distribution: {np.bincount(self.y_discrete)}")
        print(f"Grouped target distribution: {np.bincount(self.y_grouped)}")
        print(f"Class names: {class_names}")
        
        return self.X_text, self.X_numerical, self.y_grouped
    
    def create_text_features(self, max_features=1000):
        """Create text features using TF-IDF with reduced complexity"""
        print("Creating text features...")
        
        # TF-IDF Vectorizer with reduced complexity for faster training
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,  # Reduced from 5000 to 1000
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for faster processing
            min_df=5,  # Increased min_df to reduce vocabulary size
            max_df=0.9  # Slightly reduced max_df
        )
        
        # Fit and transform text data
        X_tfidf = self.tfidf_vectorizer.fit_transform(self.X_text)
        
        print(f"TF-IDF features shape: {X_tfidf.shape}")
        
        return X_tfidf
    
    def combine_features(self, X_tfidf):
        """Combine text and numerical features"""
        print("Combining features...")
        
        # Convert sparse matrix to dense for combination
        X_tfidf_dense = X_tfidf.toarray()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_numerical_scaled = self.scaler.fit_transform(self.X_numerical)
        
        # Combine features
        X_combined = np.hstack([X_tfidf_dense, X_numerical_scaled])
        
        print(f"Combined features shape: {X_combined.shape}")
        
        return X_combined
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("Splitting data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models with reduced complexity and timeout handling"""
        print("Training models...")
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        # Define models with reduced complexity for faster training
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=8,      # Reduced from 10
                min_samples_split=10,  # Added to prevent overfitting
                min_samples_leaf=5,    # Added to prevent overfitting
                random_state=42,
                n_jobs=2  # Reduced from -1 to prevent resource exhaustion
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=4,      # Reduced from 6
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=500,     # Reduced from 1000
                random_state=42,
                multi_class='ovr',
                solver='liblinear'  # Faster solver for smaller datasets
            )
        }
        
        # Train and evaluate models with timeout
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                # Train model with timeout handling
                def train_with_timeout():
                    model.fit(X_train, y_train)
                    return model
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(train_with_timeout)
                    try:
                        # 5 minute timeout per model
                        trained_model = future.result(timeout=300)
                        training_time = time.time() - start_time
                        print(f"{name} training completed in {training_time:.2f} seconds")
                        
                        # Cross-validation score with reduced CV folds for speed
                        print(f"Running cross-validation for {name}...")
                        cv_start = time.time()
                        cv_scores = cross_val_score(trained_model, X_train, y_train, cv=3, scoring='accuracy')
                        cv_time = time.time() - cv_start
                        mean_cv_score = cv_scores.mean()
                        
                        print(f"{name} - CV Accuracy: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f}) in {cv_time:.2f}s")
                        
                        # Store model and score
                        self.models[name] = {
                            'model': trained_model,
                            'cv_score': mean_cv_score,
                            'cv_std': cv_scores.std(),
                            'training_time': training_time
                        }
                        
                        # Update best model
                        if mean_cv_score > self.best_score:
                            self.best_score = mean_cv_score
                            self.best_model = name
                            
                    except TimeoutError:
                        print(f"Training {name} timed out after 5 minutes - skipping")
                        continue
                        
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if self.best_model:
            print(f"\nBest model: {self.best_model} with CV accuracy: {self.best_score:.4f}")
        else:
            print("\nNo models were successfully trained!")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        print("\nEvaluating models on test set...")
        
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'cv_score': model_info['cv_score']
                }
                
                print(f"{name} - Test Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return results
    
    def detailed_evaluation(self, X_test, y_test, model_name=None):
        """Perform detailed evaluation of a specific model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        print(f"\nDetailed evaluation for {model_name}:")
        
        model = self.models[model_name]['model']
        y_pred = model.predict(X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, model_name)
        
        return y_pred
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            return
        
        # Get feature names
        tfidf_features = [f'tfidf_{i}' for i in range(len(self.tfidf_vectorizer.get_feature_names_out()))]
        numerical_features = self.X_numerical.columns.tolist()
        all_features = tfidf_features + numerical_features
        
        # Get importance scores
        importances = model.feature_importances_
        
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        top_features = [all_features[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            model = SVC(random_state=42, probability=True)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[f'{model_name} (Tuned)'] = {
            'model': grid_search.best_estimator_,
            'cv_score': grid_search.best_score_,
            'cv_std': 0,
            'best_params': grid_search.best_params_
        }
        
        return grid_search.best_estimator_
    
    def save_model(self, model_name=None, filepath=None):
        """Save the trained model"""
        if model_name is None:
            model_name = self.best_model
        
        if filepath is None:
            filepath = f'urgency_classifier_{model_name.lower().replace(" ", "_")}.joblib'
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        # Save model, vectorizer, and scaler
        model_data = {
            'model': self.models[model_name]['model'],
            'vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'feature_names': self.X_numerical.columns.tolist(),
            'model_info': self.models[model_name]
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        print(f"Loading model from {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.loaded_model = model_data['model']
        self.loaded_vectorizer = model_data['vectorizer']
        self.loaded_scaler = model_data['scaler']
        self.loaded_feature_names = model_data['feature_names']
        
        print("Model loaded successfully!")
        return model_data
    
    def predict_urgency(self, text, numerical_features=None):
        """Predict urgency for new text"""
        if not hasattr(self, 'loaded_model'):
            print("No model loaded! Please load a model first.")
            return None
        
        # Preprocess text
        if isinstance(text, str):
            text = [text]
        
        # Vectorize text
        X_text = self.loaded_vectorizer.transform(text)
        
        # Handle numerical features
        if numerical_features is None:
            # Use default values
            numerical_features = np.zeros((len(text), len(self.loaded_feature_names)))
        else:
            numerical_features = np.array(numerical_features).reshape(len(text), -1)
        
        # Scale numerical features
        X_numerical = self.loaded_scaler.transform(numerical_features)
        
        # Combine features
        X_combined = np.hstack([X_text.toarray(), X_numerical])
        
        # Predict
        prediction = self.loaded_model.predict(X_combined)
        probability = self.loaded_model.predict_proba(X_combined) if hasattr(self.loaded_model, 'predict_proba') else None
        
        return {
            'urgency': prediction[0] if len(prediction) == 1 else prediction,
            'probability': probability[0] if probability is not None and len(probability) == 1 else probability
        }
    
    def create_model_comparison_plot(self, results):
        """Create a comparison plot of all models"""
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
        plt.bar(x + width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main(classification_type='binary', max_features=1000):
    """Main function to run the urgency classifier
    
    Args:
        classification_type: Type of classification ('binary', 'ternary', 'multiclass')
        max_features: Maximum number of TF-IDF features to use
    """
    print(f"Starting urgency classification with {classification_type} classification...")
    
    # Initialize classifier
    classifier = UrgencyClassifier(classification_type=classification_type)
    
    # Load data
    data = classifier.load_data()
    
    # Prepare features
    X_text, X_numerical, y = classifier.prepare_features()
    
    # Create text features with specified max_features
    X_tfidf = classifier.create_text_features(max_features=max_features)
    
    # Combine features
    X_combined = classifier.combine_features(X_tfidf)
    
    # Split data
    X_train, X_test, y_train, y_test = classifier.split_data(X_combined, y)
    
    # Train models
    models = classifier.train_models(X_train, y_train)
    
    # Only proceed if we have trained models
    if not models:
        print("No models were successfully trained. Exiting.")
        return classifier
    
    # Evaluate models
    results = classifier.evaluate_models(X_test, y_test)
    
    # Create comparison plot if we have results
    if results:
        classifier.create_model_comparison_plot(results)
        
        # Detailed evaluation of best model
        if classifier.best_model:
            classifier.detailed_evaluation(X_test, y_test)
            
            # Save the best model
            classifier.save_model()
    
    return classifier

if __name__ == "__main__":
    classifier = main()