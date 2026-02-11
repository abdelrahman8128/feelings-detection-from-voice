"""
Data downloader module for voice emotion detection datasets.
Handles automatic download and extraction of emotion speech datasets.
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import kaggle
from typing import Optional, List, Tuple


class EmotionDataDownloader:
    """Downloads and manages emotion speech datasets."""
    
    # Kaggle input directory where pre-mounted datasets are available
    KAGGLE_INPUT_DIR = Path("/kaggle/input")
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.is_kaggle = self._detect_kaggle_environment()
        
        # Dataset configurations
        self.datasets = {
            "ravdess": {
                "kaggle_dataset": "uwrfkaggler/ravdess-emotional-speech-audio",
                "kaggle_input_path": self.KAGGLE_INPUT_DIR / "ravdess-emotional-speech-audio",
                "local_path": self.data_dir / "ravdess",
                "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
            },
            "tess": {
                "kaggle_dataset": "ejlok1/toronto-emotional-speech-set-tess",
                "kaggle_input_path": self.KAGGLE_INPUT_DIR / "toronto-emotional-speech-set-tess",
                "local_path": self.data_dir / "tess", 
                "emotions": ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprised", "sad"]
            },
            "crema": {
                "kaggle_dataset": "ejlok1/cremad",
                "kaggle_input_path": self.KAGGLE_INPUT_DIR / "cremad",
                "local_path": self.data_dir / "crema",
                "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad"]
            }
        }
        
        if self.is_kaggle:
            print("Kaggle environment detected. Using pre-mounted datasets from /kaggle/input/")
    
    @staticmethod
    def _detect_kaggle_environment() -> bool:
        """Detect if running inside a Kaggle notebook."""
        return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle/input").exists()
    
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are properly configured."""
        try:
            kaggle.api.authenticate()
            return True
        except Exception as e:
            print(f"Kaggle authentication failed: {e}")
            print("Please ensure you have:")
            print("1. Created a Kaggle account")
            print("2. Generated API token from kaggle.com/account")
            print("3. Placed kaggle.json in ~/.kaggle/ directory")
            return False
    
    def download_from_url(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file from URL with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip file to specified directory."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_name: str, download_path: Path) -> bool:
        """Download dataset from Kaggle."""
        try:
            if not self.check_kaggle_credentials():
                return False
            
            print(f"Downloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=download_path.parent,
                unzip=True
            )
            return True
        except Exception as e:
            print(f"Error downloading Kaggle dataset {dataset_name}: {e}")
            return False
    
    def _get_dataset_search_dirs(self, dataset_key: str) -> List[Path]:
        """Get the directories to search for a dataset's audio files.
        
        On Kaggle, this includes the pre-mounted input path. Locally, it
        uses the configured data directory.
        """
        dirs = [self.data_dir]
        if self.is_kaggle and dataset_key in self.datasets:
            kaggle_path = self.datasets[dataset_key]["kaggle_input_path"]
            if kaggle_path.exists():
                dirs.append(kaggle_path)
        return dirs

    def is_dataset_available(self, dataset_key: str) -> bool:
        """Check if dataset is already downloaded and available."""
        # On Kaggle, check the pre-mounted input directory first
        if self.is_kaggle and dataset_key in self.datasets:
            kaggle_path = self.datasets[dataset_key]["kaggle_input_path"]
            if kaggle_path.exists():
                return True

        # Check in the main data directory for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.data_dir.rglob(f"*{ext}")))
        
        # For RAVDESS, check for Actor directories
        if dataset_key == "ravdess":
            actor_dirs = list(self.data_dir.glob("Actor_*"))
            return len(actor_dirs) > 0 and len(audio_files) > 0
        
        # For TESS, check for TESS directory or files
        elif dataset_key == "tess":
            tess_dir = self.data_dir / "TESS Toronto emotional speech set data"
            return tess_dir.exists() or len(audio_files) > 0
        
        # For CREMA, check for zip file or extracted files
        elif dataset_key == "crema":
            crema_zip = self.data_dir / "cremad.zip"
            return crema_zip.exists() or len(audio_files) > 0
        
        return len(audio_files) > 0
    
    def download_dataset(self, dataset_key: str) -> bool:
        """Download specific dataset if not already available."""
        if dataset_key not in self.datasets:
            print(f"Unknown dataset: {dataset_key}")
            return False
        
        if self.is_dataset_available(dataset_key):
            dataset_config = self.datasets[dataset_key]
            if self.is_kaggle and dataset_config["kaggle_input_path"].exists():
                print(f"Dataset {dataset_key} available via Kaggle input at {dataset_config['kaggle_input_path']}")
            else:
                print(f"Dataset {dataset_key} already available at {dataset_config['local_path']}")
            return True
        
        # On Kaggle, guide the user to add the dataset through the UI
        if self.is_kaggle:
            dataset_config = self.datasets[dataset_key]
            print(f"Dataset {dataset_key} not found in /kaggle/input/.")
            print(f"Please add it via the Kaggle notebook sidebar:")
            print(f"  1. Click '+ Add Data' in the right sidebar")
            print(f"  2. Search for: {dataset_config['kaggle_dataset']}")
            print(f"  3. Click 'Add' to attach it to your notebook")
            return False
        
        dataset_config = self.datasets[dataset_key]
        kaggle_dataset = dataset_config["kaggle_dataset"]
        local_path = dataset_config["local_path"]
        
        print(f"Downloading {dataset_key} dataset...")
        local_path.mkdir(parents=True, exist_ok=True)
        
        success = self.download_kaggle_dataset(kaggle_dataset, local_path)
        
        if success:
            print(f"Successfully downloaded {dataset_key} to {local_path}")
        else:
            print(f"Failed to download {dataset_key}")
        
        return success
    
    def download_all_datasets(self) -> List[str]:
        """Download all available datasets."""
        successful_downloads = []
        
        for dataset_key in self.datasets.keys():
            if self.download_dataset(dataset_key):
                successful_downloads.append(dataset_key)
        
        return successful_downloads
    
    def get_dataset_info(self) -> pd.DataFrame:
        """Get information about available datasets."""
        info_data = []
        
        for key, config in self.datasets.items():
            available = self.is_dataset_available(key)
            audio_count = 0
            
            if available:
                audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
                audio_files = []
                # Search local path
                for ext in audio_extensions:
                    audio_files.extend(list(config["local_path"].rglob(f"*{ext}")))
                # Also search Kaggle input path when running on Kaggle
                if self.is_kaggle and config["kaggle_input_path"].exists():
                    for ext in audio_extensions:
                        audio_files.extend(list(config["kaggle_input_path"].rglob(f"*{ext}")))
                audio_count = len(audio_files)
            
            effective_path = str(config["kaggle_input_path"]) if (self.is_kaggle and config["kaggle_input_path"].exists()) else str(config["local_path"])
            
            info_data.append({
                "Dataset": key.upper(),
                "Available": available,
                "Audio Files": audio_count,
                "Emotions": len(config["emotions"]),
                "Emotion Labels": ", ".join(config["emotions"]),
                "Path": effective_path
            })
        
        return pd.DataFrame(info_data)
    
    def get_file_paths_and_labels(self, dataset_keys: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """Get file paths and corresponding emotion labels from downloaded datasets."""
        if dataset_keys is None:
            dataset_keys = list(self.datasets.keys())
        
        file_paths = []
        labels = []
        
        # Search in the entire data directory for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        # Collect all directories to search
        search_dirs: List[Path] = [self.data_dir]
        if self.is_kaggle:
            for key in dataset_keys:
                if key in self.datasets:
                    kaggle_path = self.datasets[key]["kaggle_input_path"]
                    if kaggle_path.exists():
                        search_dirs.append(kaggle_path)
        
        seen_files: set = set()
        for search_dir in search_dirs:
            for ext in audio_extensions:
                audio_files = list(search_dir.rglob(f"*{ext}"))
                
                for audio_file in audio_files:
                    # Avoid duplicates when data_dir overlaps with kaggle path
                    resolved = str(audio_file.resolve())
                    if resolved in seen_files:
                        continue
                    seen_files.add(resolved)

                    # Determine which dataset this file belongs to and extract emotion
                    dataset_key = self._determine_dataset(audio_file)
                    if dataset_key in dataset_keys:
                        emotion = self._extract_emotion_from_path(audio_file, dataset_key)
                        if emotion:
                            file_paths.append(str(audio_file))
                            labels.append(emotion)
        
        return file_paths, labels
    
    def _extract_emotion_from_path(self, file_path: Path, dataset_key: str) -> Optional[str]:
        """Extract emotion label from file path based on dataset structure."""
        file_path_str = str(file_path).lower()
        
        if dataset_key == "ravdess":
            # RAVDESS filename format: 03-01-06-01-02-01-12.wav
            # Third number represents emotion (01=neutral, 02=calm, 03=happy, etc.)
            filename = file_path.stem
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 
                             5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
                return emotion_map.get(emotion_code)
        
        elif dataset_key == "tess":
            # TESS files are organized by emotion in directory structure
            for emotion in self.datasets[dataset_key]["emotions"]:
                if emotion.lower() in file_path_str:
                    return emotion.lower()
        
        elif dataset_key == "crema":
            # CREMA-D filename format includes emotion code
            for emotion in self.datasets[dataset_key]["emotions"]:
                if emotion.upper() in file_path.stem.upper():
                    return emotion.lower()
        
        # Fallback: check if any emotion is in the path
        all_emotions = ["angry", "sad", "happy", "fearful", "surprised", "disgust", "neutral", "calm"]
        for emotion in all_emotions:
            if emotion in file_path_str:
                return emotion
        
        return None
    
    def _determine_dataset(self, file_path: Path) -> str:
        """Determine which dataset a file belongs to based on its path."""
        file_path_str = str(file_path).lower()
        
        # Check for RAVDESS (Actor directories)
        if "actor_" in file_path_str:
            return "ravdess"
        
        # Check for TESS (TESS directory or specific naming)
        if "tess" in file_path_str or any(emotion in file_path_str for emotion in ["angry", "disgust", "fear", "happy", "neutral", "pleasant", "sad"]):
            return "tess"
        
        # Check for CREMA (specific naming pattern)
        if "crema" in file_path_str:
            return "crema"
        
        # Default to RAVDESS for numbered files
        return "ravdess"


def main():
    """Main function for testing the data downloader."""
    downloader = EmotionDataDownloader()
    
    print("Voice Emotion Detection - Data Downloader")
    print("=" * 50)
    
    # Show dataset information
    print("\nDataset Information:")
    print(downloader.get_dataset_info().to_string(index=False))
    
    # Download datasets
    print("\nDownloading datasets...")
    successful_downloads = downloader.download_all_datasets()
    
    if successful_downloads:
        print(f"\nSuccessfully downloaded: {', '.join(successful_downloads)}")
        
        # Get file paths and labels
        file_paths, labels = downloader.get_file_paths_and_labels()
        print(f"\nTotal audio files found: {len(file_paths)}")
        
        # Show emotion distribution
        from collections import Counter
        emotion_counts = Counter(labels)
        print("\nEmotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")
    else:
        print("No datasets were successfully downloaded.")


if __name__ == "__main__":
    main()
