#!/usr/bin/env python3
"""
Main script to run IMDB dataset preprocessing pipeline
"""

import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from data_preprocessor import IMDBPreprocessor
    print("✓ Successfully imported IMDBPreprocessor")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import...")
    # Try direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_preprocessor", "src/data_preprocessor.py")
    data_preprocessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_preprocessor)
    IMDBPreprocessor = data_preprocessor.IMDBPreprocessor

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
        print("Initializing IMDB Preprocessor...")
        preprocessor = IMDBPreprocessor(data_path)
        print("Starting full preprocessing pipeline...")
        preprocessor.run_full_pipeline()
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - cleaned_data/ : Cleaned datasets")
        print("   - splits/       : Train/test splits") 
        print("   - results/imdb/ : Charts and summary reports")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()