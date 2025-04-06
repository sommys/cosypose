from pathlib import Path

# Directory containing the files
ds_dir = Path(
    "/home/sommys/Documents/datagen/deps/cosypose/local_data/synt_datasets/ballvalve-falltest/dumps"
)

# List and sort files by splitting the key on "-" and sorting based on the two integers in the key
files = sorted(
    ds_dir.iterdir(),
    key=lambda f: (int(f.stem.split("-")[0]), int(f.stem.split("-")[1])),
)

# Extract filenames without extensions
all_keys = [f.stem for f in files]
train_ratio = 0.95
n_train = int(train_ratio * len(all_keys))
train_keys, val_keys = all_keys[:n_train], all_keys[n_train:]

# Write the keys into a txt where each line is a key
keys_txt = ds_dir.parent / "keys_recorded.txt"
keys_txt.write_text(f"{'\n'.join(all_keys)}\n")

# Collect the "seeds" which are the first part of a key before the "-" and store only unique ones
seeds = set()
for key in all_keys:
    seeds.add(key.split("-")[0])
seeds = sorted(seeds)
seeds_txt = ds_dir.parent / "seeds_recorded.txt"
seeds_txt.write_text("\n".join(seeds) + "\n")

# Write the keys into pkl files
# keys_dir = ds_dir.parent
# (keys_dir / "keys.pkl").write_bytes(pickle.dumps(all_keys))
# (keys_dir / "train_keys.pkl").write_bytes(pickle.dumps(train_keys))
# (keys_dir / "val_keys.pkl").write_bytes(pickle.dumps(val_keys))
