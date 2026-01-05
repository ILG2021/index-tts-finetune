import os
import json
import librosa
import argparse
import pandas as pd
from tqdm import tqdm


def create_manifest_ljspeech(metadata_path, wav_dir, manifest_path, lang):
    """
    Generate a NeMo manifest from LJSpeech-style metadata.
    Assumes metadata file is pipe-separated (|) with 2 columns: file_id | transcription
    """
    manifest = []

    print(f"Reading metadata from {metadata_path}...")

    # Read the metadata file. LJSpeech usually uses '|' as delimiter and no header.
    # We assume 2 columns based on user request.
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    print(f"Processing {len(lines)} entries...")

    for line in tqdm(lines):
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue

        file_id = parts[0]
        transcript = parts[1]

        # Construct audio path. LJSpeech files are usually .wav
        audio_path = os.path.join(wav_dir, f"{file_id}")

        if not os.path.exists(audio_path):
            # Try without .wav if ID already contains it, or check for common variants
            if not audio_path.endswith('.wav'):
                audio_path += '.wav'
            if not os.path.exists(audio_path):
                continue

        # Get duration
        try:
            duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue

        manifest.append({
            "audio_filepath": os.path.abspath(audio_path),
            "duration": duration,
            "text": transcript,
            "lang": lang
        })

    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nManifest created at {manifest_path} with {len(manifest)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NeMo manifest from LJSpeech folder")
    parser.add_argument("--data_folder", type=str, required=True, help="Root directory of LJSpeech dataset")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to save the manifest.json")
    parser.add_argument("--lang", type=str, default="zh", help="Language code")

    args = parser.parse_args()

    # Define common LJSpeech sub-paths
    potential_metadata = ["metadata.csv", "metadata.txt", "transcript.txt"]
    potential_wav_dirs = ["wavs", "wav", "audio", ""]
    import os
    import json
    import librosa
    import argparse
    import pandas as pd
    from tqdm import tqdm


    def create_manifest_ljspeech(metadata_path, wav_dir, manifest_path, lang):
        """
        Generate a NeMo manifest from LJSpeech-style metadata.
        Assumes metadata file is pipe-separated (|) with 2 columns: file_id | transcription
        """
        manifest = []

        print(f"Reading metadata from {metadata_path}...")

        # Read the metadata file. LJSpeech usually uses '|' as delimiter and no header.
        # We assume 2 columns based on user request.
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return

        print(f"Processing {len(lines)} entries...")

        for line in tqdm(lines):
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue

            file_id = parts[0]
            transcript = parts[1]

            # Construct audio path. LJSpeech files are usually .wav
            audio_path = os.path.join(wav_dir, f"{file_id}")

            if not os.path.exists(audio_path):
                # Try without .wav if ID already contains it, or check for common variants
                if not audio_path.endswith('.wav'):
                    audio_path += '.wav'
                if not os.path.exists(audio_path):
                    continue

            # Get duration
            try:
                duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Error processing {file_id}: {e}")
                continue

            manifest.append({
                "id": file_id,
                "audio": os.path.abspath(audio_path),
                "duration": duration,
                "text": transcript,
                "language": lang
            })

        with open(manifest_path, 'w', encoding='utf-8') as f:
            for entry in manifest:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"\nManifest created at {manifest_path} with {len(manifest)} entries.")


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Generate NeMo manifest from LJSpeech folder")
        parser.add_argument("--data_folder", type=str, required=True, help="Root directory of LJSpeech dataset")
        parser.add_argument("--output_manifest", type=str, required=True, help="Path to save the manifest.json")
        parser.add_argument("--lang", type=str, default="zh", help="Language code")

        args = parser.parse_args()

        # Define common LJSpeech sub-paths
        potential_metadata = ["metadata.csv", "metadata.txt", "transcript.txt"]
        potential_wav_dirs = ["wavs", "wav", "audio", ""]

        metadata_path = None
        for m in potential_metadata:
            path = os.path.join(args.data_folder, m)
            if os.path.exists(path):
                metadata_path = path
                break

        if not metadata_path:
            print(f"Error: Could not find metadata file in {args.data_folder}")
            exit(1)

        wav_dir = None
        for d in potential_wav_dirs:
            path = os.path.join(args.data_folder, d)
            if os.path.isdir(path):
                # Check if there are wav files here
                if any(f.endswith('.wav') for f in os.listdir(path)):
                    wav_dir = path
                    break

        if not wav_dir:
            wav_dir = os.path.join(args.data_folder, "wavs")  # Fallback to root

        create_manifest_ljspeech(metadata_path, wav_dir, args.output_manifest, args.lang)
    metadata_path = None
    for m in potential_metadata:
        path = os.path.join(args.data_folder, m)
        if os.path.exists(path):
            metadata_path = path
            break

    if not metadata_path:
        print(f"Error: Could not find metadata file in {args.data_folder}")
        exit(1)

    wav_dir = None
    for d in potential_wav_dirs:
        path = os.path.join(args.data_folder, d)
        if os.path.isdir(path):
            # Check if there are wav files here
            if any(f.endswith('.wav') for f in os.listdir(path)):
                wav_dir = path
                break

    if not wav_dir:
        wav_dir = os.path.join(args.data_folder, "wavs")  # Fallback to root

    create_manifest_ljspeech(metadata_path, wav_dir, args.output_manifest, args.lang)