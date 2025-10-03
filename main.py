#!/usr/bin/env python3
"""
Main script to run IMDB dataset preprocessing pipeline
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from src.data_preprocessor import IMDBPreprocessor

def main():
    """Main function to run the preprocessing pipeline"""
    
    data_path = 'data/IMDB_Dataset.csv'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the IMDB_Dataset.csv file is in the data/ directory")
        return
    
    try:
        # Initialize and run preprocessor
        preprocessor = IMDBPreprocessor(data_path)
        preprocessor.run_full_pipeline()
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()