#!/usr/bin/env python3
"""
Simplified runner script for the preprocessing pipeline
"""

from src.data_preprocessor import IMDBPreprocessor

def run_imdb_preprocessing():
    """Run IMDB preprocessing with error handling"""
    
    data_file = 'data/IMDB_Dataset.csv'
    
    try:
        print("🚀 Starting IMDB Sentiment Analysis Preprocessing Pipeline")
        print("=" * 70)
        
        preprocessor = IMDBPreprocessor(data_file)
        preprocessor.run_full_pipeline()
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - cleaned_data/ : Cleaned datasets")
        print("   - splits/       : Train/test splits") 
        print("   - results/imdb/ : Charts and summary reports")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_file}")
        print("   Please ensure IMDB_Dataset.csv is in the data/ directory")
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    run_imdb_preprocessing()