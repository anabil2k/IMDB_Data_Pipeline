# IMDB Sentiment Analysis Preprocessing Pipeline

A comprehensive, modular data preprocessing pipeline for IMDB movie 
review sentiment analysis. 

This project implements a complete OOP-based 
preprocessing workflow that transforms raw IMDB review data into clean, structured datasets ready for machine learning.

## 📋 Project Overview

This pipeline processes the IMDB movie reviews dataset through a series of automated steps:

- Data loading and validation
- Exploratory data analysis (EDA)
- Text cleaning and normalization
- Label encoding and class balance analysis
- Train/test splitting with stratification
- Comprehensive reporting and visualization


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)


### Installation

1. **Clone or download the project**


bash


```
git clone <repository-url>
cd IMDB_Data_Pipeline
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
2. **Install dependencies**


bash


```
pip install -r requirements.txt
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>

3. **Download IMDB DataSet from Kaagle**
    - Download `IMDB_Dataset.csv` from the following URL:
    [imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    
4. **Ensure your data file is in the correct location**

    - Place `IMDB_Dataset.csv` in the `data/` directory


### Running the Pipeline

**Option 1: Using the main script**

bash


```
python main.py
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
**Option 2: Using the simplified runner**

bash


```
python run_preprocessing.py
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
**Option 3: Direct module execution**

bash


```
python -m src.data_preprocessor
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
## 📁 Project Structure

text


```
IMDB-sentiment-preprocessing/
│
├── data/                   # Raw data directory
│   └── IMDB_Dataset.csv    # Original dataset
│
├── src/                    # Source code
│   └── data_preprocessor.py # Main preprocessing classes
│
├── cleaned_data/           # Processed datasets (generated)
│   └── imdb_cleaned.csv    # Cleaned and encoded dataset
│
├── splits/                 # Train/test splits (generated)
│   ├── imdb_train.csv      # Training dataset (80%)
│   └── imdb_test.csv       # Testing dataset (20%)
│
├── results/                # Analysis outputs (generated)
│   └── imdb/
│       ├── label_distribution.png
│       ├── class_balance.png
│       └── preprocessing_summary.txt
│
├── notebooks/              # Jupyter notebooks (optional)
├── main.py                 # Main execution script
├── run_preprocessing.py    # Simplified runner
├── requirements.txt        # Python dependencies
└── README.md              # This file
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
## 🔧 Pipeline Steps

### 1. Data Loading & Validation

- **Input**: `data/IMDB_Dataset.csv`
- **Process**: Robust CSV parsing with error handling
- **Output**: Standardized DataFrame with columns `['review', 'label']`


### 2. Data Cleaning

- Remove duplicate entries
- Handle missing values
- Filter invalid labels
- **Output**: Cleaned dataset with statistics


### 3. Text Preprocessing

- Convert to lowercase
- Expand Contractions: Convert contractions like "don't" to "do not"
- Remove HTML tags and URLs
- Remove special characters and emojis
- Normalize whitespaces
- Tokenization: Split text into individual words or tokens.
- Remove Stop Words: Filter out common words (e.g., "the", "a", "is") that add little meaning.
- Perform Lemmatization: Reduce words to their base or dictionary form (e.g., "running" -> "run", "better" -> "good").
- **Output**: `clean_text` column with processed reviews


### 4. Exploratory Data Analysis (EDA)

- Label distribution analysis
- Visualization of sentiment classes
- Statistical summaries
- **Output**: Charts and analysis reports


### 5. Label Encoding

- Convert text labels to numeric: `negative → 0`, `positive → 1`
- **Output**: `label_numeric` column


### 6. Class Balance Analysis

- Check dataset balance between positive/negative classes
- Provide imbalance handling recommendations
- **Output**: Balance analysis with visualizations


### 7. Train/Test Split

- 80/20 stratified split
- Maintains class distribution in both sets
- **Output**: Separate train and test datasets


### 8. Output Generation

- Save processed datasets
- Generate visualizations
- Create comprehensive summary report


## 📊 Output Files

### Generated Datasets









| File | Description | Size |
| --- | --- | --- |
| `cleaned_data/imdb_cleaned.csv` | Full cleaned dataset with encoded labels | ~50,000 reviews |
| `splits/imdb_train.csv` | Training set (80%) | ~40,000 reviews |
| `splits/imdb_test.csv` | Test set (20%) | ~10,000 reviews |



### Analysis Outputs









| File | Description |
| --- | --- |
| `results/imdb/label_distribution.png` | Bar chart of sentiment distribution |
| `results/imdb/class_balance.png` | Class balance visualization (bar + pie) |
| `results/imdb/preprocessing_summary.txt` | Comprehensive pipeline report |



## 🛠️ Technical Details

### Dependencies

txt


```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.60.0
wordcloud>=1.8.0
nltk>=3.7.0
IPython>=7.0.0
contractions>=0.1.73
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
### Class Architecture

The pipeline is built with a modular OOP design:

- **`IMDBPreprocessor`**: Main orchestrator class
- **`CSVLoader`**: Robust data loading with error recovery
- **`TextCleaner`**: Text preprocessing utilities
- **`DataAnalyzer`**: EDA and visualization tools


### Key Features

- **Robust Error Handling**: Multiple fallback strategies for CSV parsing
- **Modular Design**: Reusable components for different preprocessing tasks
- **Comprehensive Logging**: Detailed progress reporting
- **Visual Analytics**: Automated chart generation
- **Stratified Sampling**: Maintains class distribution in splits
- **Encoding Support**: Handles Windows/Linux encoding differences


## 📈 Dataset Information

### Original IMDB Dataset

- **Source**: IMDB movie reviews
- **Size**: ~50,000 reviews
- **Labels**: Positive/Negative sentiment
- **Format**: CSV with review text and sentiment labels


### Processed Dataset Features

- **Text**: Cleaned and normalized review text
- **Labels**: Numeric encoding (0=negative, 1=positive)
- **Split**: 80% training, 20% testing (stratified)


## 🎯 Usage Examples

### Basic Usage

python


```
from src.data_preprocessor import IMDBPreprocessor

# Initialize and run pipeline
preprocessor = IMDBPreprocessor('data/IMDB_Dataset.csv')
preprocessor.run_full_pipeline()
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
### Custom Configuration

python


```
# Custom preprocessing
preprocessor = IMDBPreprocessor('data/IMDB_Dataset.csv')

# Load and inspect data
preprocessor.load_data()
print(f"Dataset shape: {preprocessor.df.shape}")

# Custom text cleaning
preprocessor.clean_text_data()

# Custom split ratio
preprocessor.split_data(test_size=0.3, random_state=123)
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
## 🔍 Quality Assurance

### Data Quality Checks

- No duplicate reviews
- No missing values
- Valid sentiment labels only
- Consistent text encoding
- Balanced train/test splits


### Validation Metrics

- Class distribution preservation
- Text cleaning effectiveness
- Split stratification accuracy
- File integrity checks


## 🐛 Troubleshooting

### Common Issues

1. **CSV Parsing Errors**

    - Symptom: `ParserError: Expected 1 fields in line X, saw 2`
    - Solution: Remove delimiter option while loading the csv using Pandas, The pipeline includes robust error recovery
2. **Encoding Issues**

    - Symptom: Unicode encoding errors on Windows
    - Solution: Automatic fallback to compatible encodings
3. **Missing Dependencies**

    - Symptom: `ModuleNotFoundError`
    - Solution: Run `pip install -r requirements.txt`


### Debug Mode

For detailed debugging, run with traceback:

bash


```
python main.py 2>&1 | tee debug.log
```
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"></svg>
## 📝 License

This project is intended for educational and research purposes. The IMDB dataset is publicly available for academic use.

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request


## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review generated error logs
3. Open an issue with detailed error messages


## 🎓 Academic Use

This preprocessing pipeline is suitable for:

- Machine learning courses
- Sentiment analysis research
- NLP pipeline development
- Data preprocessing tutorials