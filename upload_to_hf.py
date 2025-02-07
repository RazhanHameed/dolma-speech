import os
import pandas as pd
from datasets import load_dataset, DatasetDict

def process_and_upload_datasets(base_dir="dataset", repo_name="DOLMA-speech"):
 

    languages = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for lang in languages:
        print(f"Processing {lang}...")
        metadata_path = os.path.join(base_dir, lang, "train", "metadata.csv")
        
        # Skip if metadata.csv doesn't exist
        if not os.path.exists(metadata_path):
            print(f"No metadata found for {lang}. Skipping this language...")
            continue
        
        lang_dir = os.path.join(base_dir, lang)

        # Load and process the dataset
        ds = load_dataset("audiofolder", data_dir=lang_dir)

        
        # Ensure dataset is a DatasetDict before uploading
        if not isinstance(ds, DatasetDict):
            print(f"Warning: Dataset for {lang} is not a DatasetDict. Skipping...")
            continue
        
        # Push to Hugging Face Hub
        try:
            ds.push_to_hub(f"{repo_name}", config_name=f"{lang}", private=True)
            print(f"Successfully uploaded {lang} dataset to {repo_name}/{lang}")
        except Exception as e:
            print(f"Error uploading {lang} dataset: {str(e)}")

    print("All datasets processed and uploaded.")

# Run the function
process_and_upload_datasets()
