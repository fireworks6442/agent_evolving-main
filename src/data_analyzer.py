"""
Data Analysis and Preprocessing Module for Stanford MOOC Forum Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self):
        """Initialize the data analyzer with necessary NLTK downloads"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self, csv_path, txt_path):
        """Load both CSV and TXT datasets"""
        print("Loading datasets...")
        
        # Load CSV data
        self.csv_data = pd.read_csv(csv_path)
        print(f"CSV data shape: {self.csv_data.shape}")
        
        # Load TXT data (tab-separated) with encoding handling
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                self.txt_data = pd.read_csv(txt_path, sep='\t', encoding=encoding)
                print(f"TXT data loaded successfully with {encoding} encoding")
                print(f"TXT data shape: {self.txt_data.shape}")
                break
            except UnicodeDecodeError:
                print(f"Failed to load with {encoding} encoding, trying next...")
                continue
        else:
            # If all encodings fail, try with error handling
            try:
                self.txt_data = pd.read_csv(txt_path, sep='\t', encoding='utf-8', errors='ignore')
                print("TXT data loaded with error handling (some characters may be lost)")
                print(f"TXT data shape: {self.txt_data.shape}")
            except Exception as e:
                print(f"Failed to load TXT data: {e}")
                self.txt_data = None
        
        return self.csv_data, self.txt_data
    
    def analyze_data_structure(self):
        """Analyze the structure of both datasets"""
        print("\n=== CSV DATA ANALYSIS ===")
        print("Columns:", self.csv_data.columns.tolist())
        print("\nData types:")
        print(self.csv_data.dtypes)
        print("\nMissing values:")
        print(self.csv_data.isnull().sum())
        print("\nBasic statistics:")
        print(self.csv_data.describe())
        
        print("\n=== TXT DATA ANALYSIS ===")
        print("Columns:", self.txt_data.columns.tolist())
        print("\nData types:")
        print(self.txt_data.dtypes)
        print("\nMissing values:")
        print(self.txt_data.isnull().sum())
        print("\nBasic statistics:")
        print(self.txt_data.describe())
        
    def analyze_urgency_distribution(self):
        """Analyze urgency score distributions"""
        plt.figure(figsize=(15, 5))
        
        # CSV urgency distribution
        plt.subplot(1, 3, 1)
        if 'Urgency_1_7' in self.csv_data.columns:
            self.csv_data['Urgency_1_7'].hist(bins=20, alpha=0.7)
            plt.title('CSV Urgency Distribution (Urgency_1_7)')
            plt.xlabel('Urgency Score')
            plt.ylabel('Frequency')
        
        # TXT urgency distribution
        plt.subplot(1, 3, 2)
        if 'Urgency(1-7)' in self.txt_data.columns:
            self.txt_data['Urgency(1-7)'].hist(bins=20, alpha=0.7)
            plt.title('TXT Urgency Distribution (Urgency 1-7)')
            plt.xlabel('Urgency Score')
            plt.ylabel('Frequency')
        
        # Combined comparison
        plt.subplot(1, 3, 3)
        if 'Urgency_1_7' in self.csv_data.columns and 'Urgency(1-7)' in self.txt_data.columns:
            plt.hist(self.csv_data['Urgency_1_7'].dropna(), alpha=0.5, label='CSV', bins=20)
            plt.hist(self.txt_data['Urgency(1-7)'].dropna(), alpha=0.5, label='TXT', bins=20)
            plt.title('Urgency Distribution Comparison')
            plt.xlabel('Urgency Score')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('urgency_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_text_features(self, text):
        """Extract additional text features"""
        if pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'sentiment_polarity': 0,
                'sentiment_subjectivity': 0,
                'question_marks': 0,
                'exclamation_marks': 0,
                'uppercase_ratio': 0
            }
        
        text_str = str(text)
        blob = TextBlob(text_str)
        
        # Basic text statistics
        text_length = len(text_str)
        words = text_str.split()
        word_count = len(words)
        sentence_count = len(blob.sentences)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Sentiment analysis
        sentiment = blob.sentiment
        
        # Punctuation analysis
        question_marks = text_str.count('?')
        exclamation_marks = text_str.count('!')
        
        # Uppercase ratio
        uppercase_chars = sum(1 for c in text_str if c.isupper())
        uppercase_ratio = uppercase_chars / len(text_str) if len(text_str) > 0 else 0
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'uppercase_ratio': uppercase_ratio
        }
    
    def preprocess_datasets(self):
        """Preprocess both datasets for machine learning"""
        print("Preprocessing datasets...")
        
        # Process CSV data
        if 'post_text' in self.csv_data.columns:
            print("Processing CSV data...")
            self.csv_data['cleaned_text'] = self.csv_data['post_text'].apply(self.clean_text)
            
            # Extract text features
            text_features = self.csv_data['post_text'].apply(self.extract_text_features)
            text_features_df = pd.DataFrame(text_features.tolist())
            self.csv_processed = pd.concat([self.csv_data, text_features_df], axis=1)
            
            # Handle urgency column
            if 'Urgency_1_7' in self.csv_data.columns:
                self.csv_processed['urgency'] = self.csv_data['Urgency_1_7']
            
        # Process TXT data
        if 'Text' in self.txt_data.columns:
            print("Processing TXT data...")
            self.txt_data['cleaned_text'] = self.txt_data['Text'].apply(self.clean_text)
            
            # Extract text features
            text_features = self.txt_data['Text'].apply(self.extract_text_features)
            text_features_df = pd.DataFrame(text_features.tolist())
            self.txt_processed = pd.concat([self.txt_data, text_features_df], axis=1)
            
            # Handle urgency column
            if 'Urgency(1-7)' in self.txt_data.columns:
                self.txt_processed['urgency'] = self.txt_data['Urgency(1-7)']
        
        print("Preprocessing completed!")
        
    def create_unified_dataset(self):
        """Create a unified dataset from both sources"""
        print("Creating unified dataset...")
        
        # Define common columns
        common_columns = ['text', 'urgency', 'cleaned_text', 'text_length', 'word_count', 
                         'sentence_count', 'avg_word_length', 'sentiment_polarity', 
                         'sentiment_subjectivity', 'question_marks', 'exclamation_marks', 
                         'uppercase_ratio']
        
        unified_data = []
        
        # Add CSV data
        if hasattr(self, 'csv_processed'):
            csv_subset = self.csv_processed.copy()
            if 'post_text' in csv_subset.columns:
                csv_subset['text'] = csv_subset['post_text']
            csv_subset['source'] = 'csv'
            
            # Add additional features if available
            if 'CourseType' in csv_subset.columns:
                csv_subset['course_type'] = csv_subset['CourseType']
            
            unified_data.append(csv_subset)
        
        # Add TXT data
        if hasattr(self, 'txt_processed'):
            txt_subset = self.txt_processed.copy()
            if 'Text' in txt_subset.columns:
                txt_subset['text'] = txt_subset['Text']
            txt_subset['source'] = 'txt'
            
            # Add additional features if available
            if 'CourseType' in txt_subset.columns:
                txt_subset['course_type'] = txt_subset['CourseType']
            if 'Opinion(1/0)' in txt_subset.columns:
                txt_subset['is_opinion'] = txt_subset['Opinion(1/0)']
            if 'Question(1/0)' in txt_subset.columns:
                txt_subset['is_question'] = txt_subset['Question(1/0)']
            if 'Answer(1/0)' in txt_subset.columns:
                txt_subset['is_answer'] = txt_subset['Answer(1/0)']
            if 'Sentiment(1-7)' in txt_subset.columns:
                txt_subset['sentiment_score'] = txt_subset['Sentiment(1-7)']
            if 'Confusion(1-7)' in txt_subset.columns:
                txt_subset['confusion_score'] = txt_subset['Confusion(1-7)']
        
        # Combine datasets
        if unified_data:
            self.unified_dataset = pd.concat(unified_data, ignore_index=True, sort=False)
            
            # Remove rows with missing urgency scores
            self.unified_dataset = self.unified_dataset.dropna(subset=['urgency'])
            
            # Remove rows with empty text
            self.unified_dataset = self.unified_dataset[
                (self.unified_dataset['text'].notna()) & 
                (self.unified_dataset['text'].str.strip() != '')
            ]
            
            print(f"Unified dataset shape: {self.unified_dataset.shape}")
            print(f"Urgency score range: {self.unified_dataset['urgency'].min()} - {self.unified_dataset['urgency'].max()}")
            
            return self.unified_dataset
        
        return None
    
    def save_processed_data(self, output_path='processed_data.csv'):
        """Save the processed unified dataset"""
        if hasattr(self, 'unified_dataset'):
            self.unified_dataset.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("No unified dataset to save. Run create_unified_dataset() first.")
    
    def generate_data_report(self):
        """Generate a comprehensive data analysis report"""
        if not hasattr(self, 'unified_dataset'):
            print("No unified dataset available. Run create_unified_dataset() first.")
            return
        
        print("\n=== DATA ANALYSIS REPORT ===")
        print(f"Total samples: {len(self.unified_dataset)}")
        print(f"Features: {self.unified_dataset.columns.tolist()}")
        
        # Urgency distribution
        print("\nUrgency Distribution:")
        urgency_dist = self.unified_dataset['urgency'].value_counts().sort_index()
        print(urgency_dist)
        
        # Text statistics
        print(f"\nText Statistics:")
        print(f"Average text length: {self.unified_dataset['text_length'].mean():.2f}")
        print(f"Average word count: {self.unified_dataset['word_count'].mean():.2f}")
        print(f"Average sentiment polarity: {self.unified_dataset['sentiment_polarity'].mean():.3f}")
        
        # Missing values
        print(f"\nMissing Values:")
        missing_values = self.unified_dataset.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Create visualizations
        self.create_analysis_plots()
    
    def create_analysis_plots(self):
        """Create analysis plots"""
        if not hasattr(self, 'unified_dataset'):
            return
        
        plt.figure(figsize=(20, 15))
        
        # Urgency distribution
        plt.subplot(3, 4, 1)
        self.unified_dataset['urgency'].hist(bins=20, alpha=0.7)
        plt.title('Urgency Distribution')
        plt.xlabel('Urgency Score')
        plt.ylabel('Frequency')
        
        # Text length distribution
        plt.subplot(3, 4, 2)
        self.unified_dataset['text_length'].hist(bins=50, alpha=0.7)
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        
        # Word count distribution
        plt.subplot(3, 4, 3)
        self.unified_dataset['word_count'].hist(bins=50, alpha=0.7)
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        # Sentiment polarity distribution
        plt.subplot(3, 4, 4)
        self.unified_dataset['sentiment_polarity'].hist(bins=30, alpha=0.7)
        plt.title('Sentiment Polarity Distribution')
        plt.xlabel('Polarity')
        plt.ylabel('Frequency')
        
        # Urgency vs Text Length
        plt.subplot(3, 4, 5)
        plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['text_length'], alpha=0.5)
        plt.title('Urgency vs Text Length')
        plt.xlabel('Urgency')
        plt.ylabel('Text Length')
        
        # Urgency vs Sentiment
        plt.subplot(3, 4, 6)
        plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['sentiment_polarity'], alpha=0.5)
        plt.title('Urgency vs Sentiment Polarity')
        plt.xlabel('Urgency')
        plt.ylabel('Sentiment Polarity')
        
        # Urgency vs Word Count
        plt.subplot(3, 4, 7)
        plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['word_count'], alpha=0.5)
        plt.title('Urgency vs Word Count')
        plt.xlabel('Urgency')
        plt.ylabel('Word Count')
        
        # Question marks vs Urgency
        plt.subplot(3, 4, 8)
        plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['question_marks'], alpha=0.5)
        plt.title('Urgency vs Question Marks')
        plt.xlabel('Urgency')
        plt.ylabel('Question Marks')
        
        # Correlation heatmap
        plt.subplot(3, 4, 9)
        numeric_cols = ['urgency', 'text_length', 'word_count', 'sentiment_polarity', 
                       'sentiment_subjectivity', 'question_marks', 'exclamation_marks']
        correlation_matrix = self.unified_dataset[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # Urgency by source
        plt.subplot(3, 4, 10)
        if 'source' in self.unified_dataset.columns:
            urgency_by_source = self.unified_dataset.groupby('source')['urgency'].mean()
            urgency_by_source.plot(kind='bar')
            plt.title('Average Urgency by Source')
            plt.ylabel('Average Urgency')
            plt.xticks(rotation=45)
        
        # Additional features if available
        if 'confusion_score' in self.unified_dataset.columns:
            plt.subplot(3, 4, 11)
            plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['confusion_score'], alpha=0.5)
            plt.title('Urgency vs Confusion Score')
            plt.xlabel('Urgency')
            plt.ylabel('Confusion Score')
        
        if 'sentiment_score' in self.unified_dataset.columns:
            plt.subplot(3, 4, 12)
            plt.scatter(self.unified_dataset['urgency'], self.unified_dataset['sentiment_score'], alpha=0.5)
            plt.title('Urgency vs Sentiment Score')
            plt.xlabel('Urgency')
            plt.ylabel('Sentiment Score')
        
        plt.tight_layout()
        plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run data analysis"""
    analyzer = DataAnalyzer()
    
    # Load data
    csv_data, txt_data = analyzer.load_data('Stanford.csv', 'stanfordMOOCForumPostsSet.txt')
    
    # Analyze data structure
    analyzer.analyze_data_structure()
    
    # Analyze urgency distribution
    analyzer.analyze_urgency_distribution()
    
    # Preprocess datasets
    analyzer.preprocess_datasets()
    
    # Create unified dataset
    unified_data = analyzer.create_unified_dataset()
    
    # Save processed data
    analyzer.save_processed_data()
    
    # Generate report
    analyzer.generate_data_report()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()