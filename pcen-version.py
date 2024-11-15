import torch
import os
import json
import librosa
import math
import numpy as np

# Add device configuration at the top
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLE_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
dataset_path = os.path.expanduser("genres_original")
json_path = "data.json"
def save_pcen(dataset_path, json_path, n_mels=80, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],  # Fixed typo in "labels"
        "pcen": []    # Changed from "mfcc" to "pcen"
    }

    num_sample_per_segment = int(SAMPLE_PER_TRACK / num_segments)
    expected_num_vector_per_segment = math.ceil(num_sample_per_segment / hop_length)

    # ... existing code ...

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # ... existing directory handling code ...

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                # Convert signal to tensor and move to GPU
                signal = torch.from_numpy(signal).to(device)
                
                for s in range(num_segments):
                    start_sample = num_sample_per_segment * s
                    end_sample = num_sample_per_segment + start_sample

                    # Move segment to GPU
                    segment = signal[start_sample:end_sample]
                    
                    # Convert back to CPU for librosa processing
                    segment_cpu = segment.cpu().numpy()
                    
                    # Calculate mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment_cpu,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels
                    )

                    # Convert mel_spec to tensor for GPU processing
                    mel_spec_tensor = torch.from_numpy(mel_spec).to(device)
                    
                    # Move back to CPU for PCEN calculation (as librosa doesn't support GPU)
                    mel_spec = mel_spec_tensor.cpu().numpy()
                    
                    # Apply PCEN
                    pcen_features = librosa.pcen(mel_spec, sr=sr)
                    pcen_features = pcen_features.T

                    if len(pcen_features) == expected_num_vector_per_segment:
                        data["pcen"].append(pcen_features.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

    # Save the data to JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        print(f"\nData saved successfully to {json_path}")
if __name__ == "__main__":
  save_pcen(dataset_path, json_path, num_segments=10)
