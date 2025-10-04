import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display

# Download required NLTK data (run this once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class TextCleaner:
    """Handles text cleaning operations"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning function
        """
        if not isinstance(text, str):
            return ""
        
        # Store original length for statistics
        original_len = len(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emojis and special characters (keep only letters and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization (split into words)
       # words = text.split()
        
        # Remove stopwords and lemmatize
       # cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        
        # Join the words back into a single string
       # cleaned_text = ' '.join(cleaned_words)
        
        
        # Update statistics
        cleaned_len = len(text)
        reduction = original_len - cleaned_len
        self.cleaning_stats.setdefault('total_reduction', 0)
        self.cleaning_stats['total_reduction'] += reduction
        self.cleaning_stats.setdefault('total_original_chars', 0)
        self.cleaning_stats['total_original_chars'] += original_len
        
        return text
    
    def get_cleaning_stats(self):
        """Return cleaning statistics"""
        if not self.cleaning_stats:
            return "No cleaning statistics available"
        
        avg_reduction = (self.cleaning_stats['total_reduction'] / 
                        self.cleaning_stats['total_original_chars']) * 100
        return f"Average character reduction: {avg_reduction:.2f}%"

class DataAnalyzer:
    """Handles exploratory data analysis and visualization"""
    
    def __init__(self):
        self.plots_dir = "results/imdb"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_label_distribution(self, df, dataset_name="IMDB"):
        """Plot and save label distribution"""
        label_counts = df['label'].value_counts()
        
        plt.figure(figsize=(10, 6))
        ax = label_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title(f'{dataset_name} Dataset - Label Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment Label', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(label_counts):
            ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return label_counts
    
    def plot_class_balance(self, df, label_column='label_numeric'):
        """Plot class balance analysis"""
        class_counts = df[label_column].value_counts()
        class_percentages = (class_counts / len(df)) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(['Negative (0)', 'Positive (1)'], class_counts.values, 
                      color=['lightcoral', 'lightgreen'], alpha=0.7)
        ax1.set_title('Class Distribution - Bar Chart', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Reviews', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{count}\n({class_percentages[bar.get_x() + bar.get_width()/2.]:.1f}%)',
                    ha='center', va='bottom')
        
        # Pie chart
        colors = ['lightcoral', 'lightgreen']
        ax2.pie(class_percentages, labels=['Negative (0)', 'Positive (1)'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/class_balance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_counts, class_percentages
    
    def check_class_balance(self, class_percentages, threshold=5):
        """Check if dataset is balanced"""
        is_balanced = abs(class_percentages[0] - class_percentages[1]) < threshold
        status = "Balanced" if is_balanced else "Imbalanced"
        imbalance_degree = abs(class_percentages[0] - class_percentages[1])
        
        return status, imbalance_degree

class IMDBPreprocessor:
    """Main class for IMDB dataset preprocessing pipeline"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.train_df = None
        self.test_df = None
        self.text_cleaner = TextCleaner()
        self.analyzer = DataAnalyzer()
        self.label_mapping = {'negative': 0, 'positive': 1}
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            'data',
            'cleaned_data', 
            'splits',
            'results/imdb',
            'src'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("Directory structure created successfully!")
    
    def load_data(self):
        """Load and standardize the dataset"""
        print("Loading IMDB dataset...")
        self.df = pd.read_csv(self.data_path, delimiter=';')
        
        # Standardize column names
        self.df = self.df.rename(columns={'review': 'review', 'sentiment': 'label'})
        
        print(f"Original dataset size: {self.df.shape}")
        return self.df.shape
    
    def clean_data(self):
        """Remove duplicates and missing values"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove duplicates
        self.df_clean = self.df.drop_duplicates()
        
        # Remove missing values
        self.df_clean = self.df_clean.dropna()
        
        original_size = self.df.shape[0]
        cleaned_size = self.df_clean.shape[0]
        removed_rows = original_size - cleaned_size
        
        print(f"Cleaned dataset size: {self.df_clean.shape}")
        print(f"Removed rows: {removed_rows}")
        
        return original_size, cleaned_size, removed_rows
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")
        
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Show random samples
        print("5 Random Samples:")
        display(self.df_clean.sample(5))
        
        # Plot label distribution
        label_counts = self.analyzer.plot_label_distribution(self.df_clean)
        
        # Calculate percentages
        total_reviews = len(self.df_clean)
        positive_pct = (label_counts['positive'] / total_reviews) * 100
        negative_pct = (label_counts['negative'] / total_reviews) * 100
        
        print(f"\nDataset has {positive_pct:.2f}% positive and {negative_pct:.2f}% negative reviews.")
        
        return label_counts, positive_pct, negative_pct
    
    def clean_text_data(self):
        """Apply text cleaning to reviews"""
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")
        
        print("\n=== TEXT CLEANING ===")
        print("Cleaning text data...")
        
        self.df_clean['clean_text'] = self.df_clean['review'].apply(self.text_cleaner.clean_text)
        
        # Show cleaning examples
        print("\nText Cleaning Examples:")
        sample_indices = self.df_clean.sample(3).index
        for idx in sample_indices:
            original = self.df_clean.loc[idx, 'review'][:150] + "..." 
            cleaned = self.df_clean.loc[idx, 'clean_text'][:150] + "..."
            print(f"\nOriginal: {original}")
            print(f"Cleaned:  {cleaned}")
            print("-" * 50)
        
        print(self.text_cleaner.get_cleaning_stats())
    
    def encode_labels(self):
        """Convert labels to numeric format"""
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")
        
        print("\n=== LABEL ENCODING ===")
        self.df_clean['label_numeric'] = self.df_clean['label'].map(self.label_mapping)
        
        # Verify encoding
        unmapped = self.df_clean[self.df_clean['label_numeric'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: {len(unmapped)} unmapped labels found")
        else:
            print("All labels successfully mapped.")
        
        print("Label mapping:", self.label_mapping)
        print("\nFirst 5 rows with numeric labels:")
        print(self.df_clean[['review', 'label', 'label_numeric']].head())
    
    def analyze_class_balance(self):
        """Analyze and visualize class balance"""
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")
        
        print("\n=== CLASS BALANCE ANALYSIS ===")
        
        class_counts, class_percentages = self.analyzer.plot_class_balance(self.df_clean)
        status, imbalance_degree = self.analyzer.check_class_balance(class_percentages)
        
        # Create balance table
        balance_table = pd.DataFrame({
            'Class': ['Negative (0)', 'Positive (1)'],
            'Count': class_counts.values,
            'Percentage': class_percentages.values
        })
        
        print("Class Distribution:")
        print(balance_table)
        print(f"\nDataset Status: {status}")
        
        if status == "Imbalanced":
            print(f"Imbalance degree: {imbalance_degree:.2f}%")
            self._suggest_imbalance_strategies()
        
        return balance_table, status, imbalance_degree
    
    def _suggest_imbalance_strategies(self):
        """Suggest strategies for handling class imbalance"""
        print("\nRecommended strategies for class imbalance:")
        print("1. Class weights in model training")
        print("2. Oversampling techniques (SMOTE)")
        print("3. Undersampling majority class")
        print("4. Data augmentation for minority class")
        print("5. Collect more data for minority class")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")
        
        print("\n=== DATA SPLITTING ===")
        
        X = self.df_clean['clean_text']
        y = self.df_clean['label_numeric']
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Create DataFrames
        self.train_df = pd.DataFrame({
            'clean_text': X_train,
            'label': y_train
        })
        
        self.test_df = pd.DataFrame({
            'clean_text': X_test,
            'label': y_test
        })
        
        # Print split statistics
        train_size = len(self.train_df)
        test_size = len(self.test_df)
        
        print(f"Train set size: {train_size} ({train_size/len(self.df_clean)*100:.1f}%)")
        print(f"Test set size: {test_size} ({test_size/len(self.df_clean)*100:.1f}%)")
        
        # Check class distribution in splits
        train_positive_pct = (self.train_df['label'].sum() / len(self.train_df)) * 100
        test_positive_pct = (self.test_df['label'].sum() / len(self.test_df)) * 100
        
        split_summary = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'Size': [train_size, test_size],
            'Positive %': [f"{train_positive_pct:.2f}%", f"{test_positive_pct:.2f}%"],
            'Negative %': [f"{100-train_positive_pct:.2f}%", f"{100-test_positive_pct:.2f}%"]
        })
        
        print("\nSplit Summary:")
        print(split_summary)
        
        return split_summary
    
    def save_data(self):
        """Save all processed data"""
        if self.df_clean is None:
            raise ValueError("No data to save. Run preprocessing first.")
        
        print("\n=== SAVING DATA ===")
        
        # Save cleaned dataset
        self.df_clean.to_csv('cleaned_data/imdb_cleaned.csv', index=False)
        print("✓ Cleaned dataset saved to 'cleaned_data/imdb_cleaned.csv'")
        
        # Save split datasets
        if self.train_df is not None and self.test_df is not None:
            self.train_df.to_csv('splits/imdb_train.csv', index=False)
            self.test_df.to_csv('splits/imdb_test.csv', index=False)
            print("✓ Train set saved to 'splits/imdb_train.csv'")
            print("✓ Test set saved to 'splits/imdb_test.csv'")
    
    def generate_summary_report(self, original_size, cleaned_size, removed_rows, 
                              label_counts, balance_table, status, split_summary):
        """Generate comprehensive summary report"""
        print("\n=== GENERATING SUMMARY REPORT ===")
        
        positive_pct = (label_counts['positive'] / len(self.df_clean)) * 100
        negative_pct = (label_counts['negative'] / len(self.df_clean)) * 100
        
        summary = f"""
IMDB SENTIMENT ANALYSIS - PREPROCESSING SUMMARY
===============================================

DATASET OVERVIEW:
• Original size: {original_size} reviews
• After cleaning: {cleaned_size} reviews
• Removed: {removed_rows} rows (duplicates/missing values)

LABEL DISTRIBUTION:
• Positive reviews: {label_counts['positive']} ({positive_pct:.2f}%)
• Negative reviews: {label_counts['negative']} ({negative_pct:.2f}%)
• Dataset status: {status}

TRAIN/TEST SPLIT:
• Train set: {split_summary.iloc[0]['Size']} reviews
• Test set: {split_summary.iloc[1]['Size']} reviews
• Stratification: Yes (maintains class distribution)

PREPROCESSING STEPS COMPLETED:
1. ✓ Data loading and column standardization
2. ✓ Duplicate and missing value removal
3. ✓ Text cleaning (lowercase, HTML removal, special characters, etc.)
4. ✓ Label encoding (negative=0, positive=1)
5. ✓ Class balance analysis
6. ✓ Stratified train/test split (80/20)

FILES GENERATED:
• cleaned_data/imdb_cleaned.csv - Full cleaned dataset
• splits/imdb_train.csv - Training set
• splits/imdb_test.csv - Test set
• results/imdb/label_distribution.png - Label distribution chart
• results/imdb/class_balance.png - Class balance visualization

{'IMBALANCE HANDLING RECOMMENDATION:' if status == 'Imbalanced' else 'DATASET IS BALANCED:'}
{self._get_imbalance_recommendation(status) if status == 'Imbalanced' else 'No special handling needed for class imbalance.'}
"""
        print(summary)
        
        # Save summary to file
        with open('results/imdb/preprocessing_summary.txt', 'w') as f:
            f.write(summary)
        print("✓ Summary report saved to 'results/imdb/preprocessing_summary.txt'")
    
    def _get_imbalance_recommendation(self, status):
        """Get imbalance handling recommendations"""
        if status == "Imbalanced":
            return """
The dataset shows class imbalance. Consider using:
• Class weights in model training
• Oversampling techniques (SMOTE)
• Data augmentation for minority class
• Appropriate evaluation metrics (F1-score, precision, recall)
"""
        return ""
    
    def run_full_pipeline(self):
        """Execute complete preprocessing pipeline"""
        print("Starting IMDB Dataset Preprocessing Pipeline...")
        print("=" * 60)
        
        # Step 1: Load data
        original_size = self.load_data()
        
        # Step 2: Clean data
        orig_size, cleaned_size, removed_rows = self.clean_data()
        
        # Step 3: EDA
        label_counts, positive_pct, negative_pct = self.perform_eda()
        
        # Step 4: Text cleaning
        self.clean_text_data()
        
        # Step 5: Label encoding
        self.encode_labels()
        
        # Step 6: Class balance analysis
        balance_table, status, imbalance_degree = self.analyze_class_balance()
        
        # Step 7: Data splitting
        split_summary = self.split_data()
        
        # Step 8: Save data
        self.save_data()
        
        # Step 9: Generate report
        self.generate_summary_report(
            original_size[0], cleaned_size, removed_rows,
            label_counts, balance_table, status, split_summary
        )
        
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = IMDBPreprocessor('data/IMDB_Dataset.csv')
    
    # Run full pipeline
    preprocessor.run_full_pipeline()