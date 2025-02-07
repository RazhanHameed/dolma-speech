import pandas as pd
import os
import shutil
from pathlib import Path

# Language code mappings
LANG_CODES = {
    'gilaki': 'GLK',
    'hawrami': 'HAC',
    'laki_kurdish': 'LKI',
    'mazanderani': 'MZN',
    'southern_kurdish': 'SDH',
    'talysh': 'TLY',
    'zazaki': 'ZZA'
}

def process_dataset(dataset_dir, mt_eval_dir):
    # Process each language directory
    for lang_dir in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, lang_dir)):
            continue
            
        print(f"Processing {lang_dir}...")
        
        # Get language code
        lang_code = LANG_CODES.get(lang_dir)
        if not lang_code:
            print(f"Skipping {lang_dir} - no matching language code")
            continue
            
        # Read TSV file
        tsv_path = os.path.join(mt_eval_dir, f"{lang_code}-test.tsv")
        if not os.path.exists(tsv_path):
            print(f"TSV file not found for {lang_dir}")
            continue
            
        tsv_data = pd.read_csv(tsv_path, sep='\t')
        test_sentences = set(tsv_data['fa_sentence'].str.strip())
        
        # Read metadata
        metadata_path = os.path.join(dataset_dir, lang_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"metadata.csv not found for {lang_dir}")
            continue
            
        metadata = pd.read_csv(metadata_path)
        
        # Create train and test directories
        train_dir = os.path.join(dataset_dir, lang_dir, "train")
        test_dir = os.path.join(dataset_dir, lang_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Split data
        test_mask = metadata['sentence'].str.strip().isin(test_sentences)
        test_data = metadata[test_mask]
        train_data = metadata[~test_mask]
        
        # Save metadata files
        test_data.to_csv(os.path.join(test_dir, "metadata.csv"), index=False)
        train_data.to_csv(os.path.join(train_dir, "metadata.csv"), index=False)
        
        # Copy audio files
        for idx, row in metadata.iterrows():
            src_file = os.path.join(dataset_dir, lang_dir, row['file_name'])
            if not os.path.exists(src_file):
                print(f"Warning: Audio file not found: {src_file}")
                continue
                
            dest_dir = test_dir if test_mask.iloc[idx] else train_dir
            dest_file = os.path.join(dest_dir, row['file_name'])
            shutil.move(src_file, dest_file)
            
        print(f"Completed {lang_dir}: {len(test_data)} test samples, {len(train_data)} train samples")

if __name__ == "__main__":
    dataset_dir = "dataset"
    mt_eval_dir = "mt_evaluations"
    process_dataset(dataset_dir, mt_eval_dir) 