# Create directory structure
import os

directories = [
    'data',
    'cleaned_data', 
    'splits',
    'results/imdb',
    'notebooks',
    'src'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

print("\nRepository structure created successfully!")