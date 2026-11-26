"""
Binary Urgency Classifier Module
Optimized for binary classification (Low/High urgency) with 500 TF-IDF features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BinaryUrgencyClassifier:
    def __init__(self, data_path='processed_data.csv', max_features=500, models_dir='models'):
        """Initialize the binary urgency classifier"""
        self.data_path = data_path
        self.max_features = max_features
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.tfidf_vectorizer = None
        self.scaler = None
        
        # Add logging control for fallback messages
        self._fallback_logged = False
        self._fallback_count = 0
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def get_model_path(self):
        """Get the standard model file path"""
        return os.path.join(self.models_dir, 'binary_urgency_classifier.joblib')
    
    def get_metadata_path(self):
        """Get the model metadata file path"""
        return os.path.join(self.models_dir, 'binary_urgency_classifier_metadata.json')
    
    def model_exists(self):
        """Check if a saved model exists"""
        return os.path.exists(self.get_model_path())
    
    def should_retrain(self, force_retrain=False, max_age_days=30):
        """Determine if model should be retrained"""
        if force_retrain:
            return True
            
        if not self.model_exists():
            return True
            
        # Check model age
        metadata_path = self.get_metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                training_date = datetime.fromisoformat(metadata['training_date'])
                age_days = (datetime.now() - training_date).days
                
                if age_days > max_age_days:
                    print(f"Model is {age_days} days old, retraining...")
                    return True
                    
            except Exception as e:
                print(f"Error reading metadata: {e}, retraining...")
                return True
        
        return False
        
    def load_data(self):
        """Load and prepare data for binary classification"""
        print("Loading preprocessed data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        
        # Remove rows with missing text or urgency
        self.data = self.data.dropna(subset=['text', 'urgency'])
        print(f"Data shape after removing missing values: {self.data.shape}")
        
        return self.data
    
    def prepare_binary_features(self):
        """Prepare features for binary classification (Low/High urgency)"""
        print("Preparing binary classification features...")
        
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
        
        # Binary target: 1-3 = Low (0), 4-7 = High (1)
        self.y_original = self.data['urgency'].values
        self.y_discrete = np.round(self.y_original).astype(int)
        
        # Convert to binary classification
        self.y_binary = np.where(self.y_discrete <= 3, 0, 1)  # 0=Low, 1=High
        
        print(f"Text features shape: {self.X_text.shape}")
        print(f"Numerical features shape: {self.X_numerical.shape}")
        print(f"Binary target distribution: Low={np.sum(self.y_binary == 0)}, High={np.sum(self.y_binary == 1)}")
        
        return self.X_text, self.X_numerical, self.y_binary
    
    def create_text_features(self):
        """Create text features using TF-IDF with specified max_features"""
        print(f"Creating text features with {self.max_features} features...")
        
        # TF-IDF Vectorizer optimized for binary classification
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for speed
            min_df=3,
            max_df=0.9,
            lowercase=True,
            strip_accents='ascii'
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
    
    def train_binary_models(self, X_train, y_train):
        """Train optimized models for binary classification"""
        print("Training binary classification models...")
        
        # Optimized models for binary classification
        models = {
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        }
        
        # Train and evaluate models with timeout
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_cv_score = cv_scores.mean()
                
                training_time = time.time() - start_time
                print(f"{name} - CV Accuracy: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f}) - Time: {training_time:.1f}s")
                
                # Store model and score
                self.models[name] = {
                    'model': model,
                    'cv_score': mean_cv_score,
                    'cv_std': cv_scores.std(),
                    'training_time': training_time
                }
                
                # Update best model
                if mean_cv_score > self.best_score:
                    self.best_score = mean_cv_score
                    self.best_model = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        print(f"\nBest model: {self.best_model} with CV accuracy: {self.best_score:.4f}")
        
        return self.models
    
    def evaluate_binary_models(self, X_test, y_test):
        """Evaluate binary classification models"""
        print("\nEvaluating binary classification models...")
        
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'cv_score': model_info['cv_score'],
                    'training_time': model_info['training_time']
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return results
    
    def detailed_evaluation(self, X_test, y_test, model_name=None):
        """Perform detailed evaluation of the best model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        print(f"\nDetailed evaluation for {model_name}:")
        
        model = self.models[model_name]['model']
        y_pred = model.predict(X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Urgency', 'High Urgency']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Urgency', 'High Urgency'],
                   yticklabels=['Low Urgency', 'High Urgency'])
        plt.title(f'Binary Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'binary_confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred
    
    def save_binary_model(self, filepath=None, performance_metrics=None):
        """Save the best binary classification model with metadata"""
        if filepath is None:
            filepath = self.get_model_path()
        
        if self.best_model not in self.models:
            print(f"Best model {self.best_model} not found!")
            return
        
        # Save model, vectorizer, and scaler
        model_data = {
            'model': self.models[self.best_model]['model'],
            'vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'feature_names': self.X_numerical.columns.tolist() if hasattr(self, 'X_numerical') else [],
            'model_info': self.models[self.best_model],
            'max_features': self.max_features,
            'classification_type': 'binary',
            'best_model_name': self.best_model,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Binary model saved to {filepath}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_name': self.best_model,
            'best_score': float(self.best_score) if hasattr(self, 'best_score') else None,
            'max_features': self.max_features,
            'data_shape': list(self.data.shape) if hasattr(self, 'data') else None,
            'performance_metrics': self._serialize_metrics(performance_metrics) if performance_metrics else {},
            'model_path': filepath
        }
        
        metadata_path = self.get_metadata_path()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")
        return filepath
    
    def load_binary_model(self, filepath=None):
        """Load a saved binary classification model"""
        if filepath is None:
            filepath = self.get_model_path()
            
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return None
            
        print(f"Loading binary model from {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.loaded_model = model_data['model']
            self.loaded_vectorizer = model_data['vectorizer']
            self.loaded_scaler = model_data['scaler']
            self.loaded_feature_names = model_data.get('feature_names', [])
            self.max_features = model_data.get('max_features', 500)
            self.best_model = model_data.get('best_model_name', 'Unknown')
            
            # Load metadata if available
            metadata_path = self.get_metadata_path()
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"Model trained on: {metadata.get('training_date', 'Unknown')}")
                    print(f"Best score: {metadata.get('best_score', 'Unknown')}")
            
            print("Binary model loaded successfully!")
            return model_data
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def train_or_load_model(self, force_retrain=False):
        """Train a new model or load existing one"""
        if not self.should_retrain(force_retrain):
            print("Loading existing model...")
            model_data = self.load_binary_model()
            if model_data is not None:
                return model_data
            else:
                print("Failed to load model, training new one...")
        
        print("Training new model...")
        return self.train_full_pipeline()
    
    def train_full_pipeline(self):
        """Complete training pipeline with automatic saving"""
        # Load and prepare data
        data = self.load_data()
        X_text, X_numerical, y_binary = self.prepare_binary_features()
        
        # Create features
        X_tfidf = self.create_text_features()
        X_combined = self.combine_features(X_tfidf)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train models
        self.train_binary_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_binary_models(X_test, y_test)
        
        # Save the best model automatically
        self.save_binary_model(performance_metrics=results)
        
        # Set the loaded model attributes so the model is immediately available
        self.loaded_model = self.models[self.best_model]['model']
        self.loaded_vectorizer = self.tfidf_vectorizer
        self.loaded_scaler = self.scaler
        self.loaded_feature_names = self.feature_names if hasattr(self, 'feature_names') else []
        
        print(f"✅ Binary model trained and loaded successfully! Best model: {self.best_model}")
        
        return {
            'model_data': {
                'model': self.models[self.best_model]['model'],
                'vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler
            },
            'results': results
        }

    def _rule_based_prediction(self, text):
        """Simple rule-based fallback prediction when model is not available"""
        if isinstance(text, str):
            text = [text]
        
        # Simple rule-based urgency detection
        urgent_keywords = [
            'urgent', 'emergency', 'asap', 'immediately', 'critical', 'deadline',
            'help', 'problem', 'issue', 'error', 'bug', 'broken', 'fail', 'crash',
            'stuck', 'confused', 'lost', 'trouble', 'difficulty', 'struggling'
        ]
        
        predictions = []
        for t in text:
            t_lower = t.lower()
            # Count urgent keywords
            urgent_count = sum(1 for keyword in urgent_keywords if keyword in t_lower)
            
            # Simple heuristic: if text contains urgent keywords or is short (likely a question)
            if urgent_count > 0 or len(t.split()) < 10:
                prediction = 1  # High urgency
                confidence = min(0.6 + urgent_count * 0.1, 0.9)  # Base confidence + keyword boost
            else:
                prediction = 0  # Low urgency
                confidence = 0.7
            
            predictions.append({
                'urgency_class': prediction,
                'urgency_label': 'High Urgency' if prediction == 1 else 'Low Urgency',
                'confidence': [1-confidence, confidence] if prediction == 1 else [confidence, 1-confidence]
            })
        
        # Return single result if single input, otherwise return list
        return predictions[0] if len(predictions) == 1 else predictions

    def predict_urgency(self, text, numerical_features=None):
        """Predict urgency for new text (binary classification)"""
        if not hasattr(self, 'loaded_model'):
            # Only log the fallback message once, then show periodic updates
            if not self._fallback_logged:
                print("⚠️  Binary model not available - using rule-based fallback for predictions")
                self._fallback_logged = True
            
            self._fallback_count += 1
            # Show progress every 50 predictions to avoid spam
            if self._fallback_count % 50 == 0:
                print(f"   📊 Rule-based predictions: {self._fallback_count}")
            
            return self._rule_based_prediction(text)
        
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
        
        # Convert to human-readable labels
        urgency_labels = ['Low Urgency', 'High Urgency']
        
        result = {
            'urgency_class': prediction[0] if len(prediction) == 1 else prediction,
            'urgency_label': urgency_labels[prediction[0]] if len(prediction) == 1 else [urgency_labels[p] for p in prediction],
            'confidence': probability[0] if probability is not None and len(probability) == 1 else probability
        }
        
        return result

    def _serialize_metrics(self, metrics):
        """Convert numpy types to JSON-serializable Python types"""
        if not metrics:
            return {}
        
        serialized = {}
        for model_name, model_metrics in metrics.items():
            serialized[model_name] = {}
            for key, value in model_metrics.items():
                if isinstance(value, np.ndarray):
                    serialized[model_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serialized[model_name][key] = float(value)
                else:
                    serialized[model_name][key] = value
        
        return serialized

def main():
    """Main function to train and evaluate binary urgency classifier"""
    # Initialize classifier with 500 features
    classifier = BinaryUrgencyClassifier(max_features=500)
    
    # Load data
    data = classifier.load_data()
    
    # Prepare binary features
    X_text, X_numerical, y_binary = classifier.prepare_binary_features()
    
    # Create text features
    X_tfidf = classifier.create_text_features()
    
    # Combine features
    X_combined = classifier.combine_features(X_tfidf)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train models
    models = classifier.train_binary_models(X_train, y_train)
    
    # Evaluate models
    results = classifier.evaluate_binary_models(X_test, y_test)
    
    # Detailed evaluation of best model
    classifier.detailed_evaluation(X_test, y_test)
    
    # Save the best model
    model_path = classifier.save_binary_model()
    
    print(f"\n=== Binary Classification Results Summary ===")
    print(f"Best Model: {classifier.best_model}")
    print(f"CV Accuracy: {classifier.best_score:.4f}")
    print(f"Test Accuracy: {results[classifier.best_model]['accuracy']:.4f}")
    print(f"Test F1-Score: {results[classifier.best_model]['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()