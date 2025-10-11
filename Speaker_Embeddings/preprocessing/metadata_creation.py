from pathlib import Path
import json
import pandas as pd
import re
# creation of metadata.csv file 
base_dir = Path("/kaggle/input/voxceleb/vox1_test_wav/wav")
if not base_dir.exists():
    for candidate in Path("/kaggle/input").iterdir():
        if any(p.name.startswith("id") for p in candidate.rglob("*") if p.is_dir()):
            base_dir = candidate
            break
            
# figures whivh .wav file belongs to which speaker
def infer_speaker_id(path: Path, base_dir: Path):
    parts = path.relative_to(base_dir).parts
    for part in parts:
        if re.match(r"^id\d+$", part):
            return part
    return parts[0]
    
# creating the table which hss three columns 1st is wav_path
# 2nd is speaker(id10270 to id10309),
# 3rd is speaker index(0 to 39) so 40 speakers 
wav_paths = sorted([p for p in base_dir.rglob("*.wav")])
rows = [{"wav_path": str(p), "speaker": infer_speaker_id(p, base_dir)} for p in wav_paths]
df = pd.DataFrame(rows)
speakers = sorted(df["speaker"].unique())
spk2idx = {s:i for i,s in enumerate(speakers)}
df["speaker_idx"] = df["speaker"].map(spk2idx)

df.to_csv("/kaggle/working/voxceleb_metadata.csv", index=False)
with open("/kaggle/working/speaker2idx.json", "w") as f:
    json.dump(spk2idx, f)
print("Metadata saved. Speakers:", len(spk2idx))