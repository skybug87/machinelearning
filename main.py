import subprocess
import sys
import os

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(result.returncode)

def main():
    scripts = [
        ("scripts/data_collection/download_and_filter.py", "download"),
        ("scripts/data_collection/convert_to_wav.py", None),
        ("scripts/preprocessing/noise_reduction.py", None),
        ("scripts/preprocessing/highpass_filter.py", None),
        ("scripts/feature_extraction/extract_features.py", None),
        ("scripts/feature_extraction/normalize_features.py", None),
    ]

    # Prompt for dataset download
    download_script, _ = scripts[0]
    prompt = "Do you want to download the dataset? (y/n): "
    answer = input(prompt).strip().lower()
    if answer == "y":
        run_script(download_script)
    else:
        print("Skipping dataset download.")

    # Run the rest of the scripts
    for script, tag in scripts[1:]:
        run_script(script)

if __name__ == "__main__":
    main()