import subprocess
import sys
import os

SCRIPTS = [
    "1_dataset_info.py",
    "2_data_visualization_and_filtering.py",
    "3_audio_processing.py",
    "4_build_spectrograms.py",
    "5_normalize_features.py",
    "6_create_splits.py",
    "7_deep_learning_pipeline.py",
    "8_traditional_pipeline.py",
]

def main():
    for script in SCRIPTS:
        script_path = os.path.join("scripts", script)
        print("="*50)
        print(f"Running {script}...")
        print("="*50)
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            print(f"Error: {script} failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
        print(f"{script} completed successfully.\n")
    print("All pipeline stages completed successfully.")

if __name__ == "__main__":
    main()