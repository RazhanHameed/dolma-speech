import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset

from seamless_communication.models.unit_extractor import UnitExtractor
from seamless_communication.datasets.huggingface import (
    SpeechTokenizer,
)
from dolma_hf import Speech2SpeechDOLMADatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")

language_mapping = {
    "english": "eng",
    "hawrami": "ckb",        # Central Kurdish (closest to Hawrami/Gorani)
    "mazanderani": "pes",    # Western Persian (closest to Mazanderani)
    "talysh": "pbt",         # Southern Pashto (another Iranian language)
    "gilaki": "arb",         # Modern Standard Arabic (geographically adjacent)
    "laki_kurdish": "urd",   # Urdu (has Persian influence)
    "southern_kurdish": "ary", # Moroccan Arabic (has similar phonology)
    "zazaki": "arz"          # Egyptian Arabic (shares some features)
}

from json import JSONEncoder

class TensorEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


class UnitSpeechTokenizer(SpeechTokenizer):
    MODEL_NAME = "xlsr2_1b_v2"
    KMEANS_MODEL_URI = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    OUTPUT_LAYER_IDX = 34

    def __init__(self, device: torch.device):
        self.device = device
        self.unit_extractor = UnitExtractor(
            model_name_or_card=self.MODEL_NAME,
            kmeans_uri=self.KMEANS_MODEL_URI,
            device=self.device,
        )

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.unit_extractor.predict(
            wav.to(self.device),
            out_layer_idx=self.OUTPUT_LAYER_IDX,
            sample_rate=sample_rate,
        )

def load_english_metadata(save_directory: str) -> pd.DataFrame:
    """Load English metadata from HuggingFace dataset."""
    logger.info("Loading English metadata...")
    
    # Load English split from dataset
    dataset = load_dataset("razhan/dolma-speech", "english", split="train", cache_dir=save_directory)
    metadata = dataset.to_pandas()
    
    logger.info(f"Loaded metadata for {len(metadata)} English samples")
    return metadata

def get_language_code(language: str) -> str:
    """Convert full language name to language code."""
    # if language.lower() == "english":
    #     return "eng"
    return language_mapping.get(language.lower(), language.lower())

def process_language_directory(
    source_lang: str,
    target_lang: str, 
    split: str,
    save_directory: Path,
    tokenizer: UnitSpeechTokenizer,
) -> None:
    """Process a specific language directory and create manifest file."""
    # dataset = load_dataset("razhan/dolma-speech", source_lang, split=split, cache_dir=save_directory)
    # Convert language names to codes
    source_lang_code = get_language_code(source_lang)
    target_lang_code = get_language_code(target_lang)
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )
    tokenizer = UnitSpeechTokenizer(device=device)
    dataset_iterator = Speech2SpeechDOLMADatasetBuilder(
        source_lang=source_lang,
        target_lang=target_lang,
        dataset_cache_dir=save_directory,
        speech_tokenizer=tokenizer,
        skip_source_audio=True,  # don't extract units from source audio
        skip_target_audio=False,
        split=split,
    )
    

    logger.info(f"Processing {source_lang} language...")

    # Create output directory if it doesn't exist 
    save_directory.mkdir(parents=True, exist_ok=True)
    manifest_path: str = os.path.join(save_directory, f"{split}_manifest.json")
    with open(manifest_path, "w") as fp_out:
        for idx, sample in enumerate(dataset_iterator.__iter__(), start=1):
           
            sample.source.lang = source_lang_code
            sample.target.lang = target_lang_code
            # sample.target.waveform = None  # already extracted units
            fp_out.write(json.dumps(dataclasses.asdict(sample),  cls=TensorEncoder) + "\n")
    logger.info(f"Saved {idx} samples for split={split} to {manifest_path}")
    logger.info(f"Manifest saved to: {manifest_path}")

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process custom dataset directory structure and create manifest files"
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language code (subdirectory name)",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language code (usually 'eng' for English)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split name (train/validation/test)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where manifest files will be saved",
    )
    return parser

def main() -> None:
    args = init_parser().parse_args()
    
    # Set up device
    device = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    tokenizer = UnitSpeechTokenizer(device=device)
    
    # Process the dataset
    process_language_directory(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        split=args.split,
        save_directory=args.save_dir,
        tokenizer=tokenizer,
    )

if __name__ == "__main__":
    main()