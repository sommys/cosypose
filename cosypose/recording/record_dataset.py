import yaml
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from .record_chunk import record_chunk


def process_seed(args):
    ds_dir, seed, n_frames_per_chunk, scene_kwargs = args
    keys, _ = record_chunk(
        ds_dir=ds_dir,
        seed=seed,
        n_frames=n_frames_per_chunk,
        scene_kwargs=scene_kwargs,
    )
    return keys, seed


def record_dataset_local(
    ds_dir,
    scene_kwargs,
    n_chunks,
    n_frames_per_chunk,
    start_seed=0,
    resume=False,
    n_workers=4,
):
    seeds = set(range(start_seed, start_seed + n_chunks))
    if resume:
        done_seeds = (ds_dir / "seeds_recorded.txt").read_text().strip().split("\n")
        seeds = set(seeds) - set(map(int, done_seeds))
        all_keys = (ds_dir / "keys_recorded.txt").read_text().strip().split("\n")
    else:
        all_keys = []
    seeds = list(seeds)

    # with multiprocessing.Pool(n_workers) as pool:
    #     results = list(
    #         tqdm(
    #             pool.imap(
    #                 process_seed,
    #                 [
    #                     (ds_dir, seed, n_frames_per_chunk, scene_kwargs)
    #                     for seed in seeds
    #                 ],
    #             ),
    #             total=len(seeds),
    #         )
    #     )
    results = []
    for seed in tqdm(seeds, desc="Processing seeds"):
        keys, _ = process_seed((ds_dir, seed, n_frames_per_chunk, scene_kwargs))
        results.append((keys, seed))

    seeds_file = open(  # pylint: disable=unspecified-encoding
        ds_dir / "seeds_recorded.txt", "a"
    )
    keys_file = open(  # pylint: disable=unspecified-encoding
        ds_dir / "keys_recorded.txt", "a"
    )

    for keys, seed in results:
        all_keys += keys
        seeds_file.write(f"{seed}\n")
        keys_file.write("\n".join(keys) + "\n")

    seeds_file.close()
    keys_file.close()
    return all_keys


def record_dataset(args):
    if args.resume and not args.overwrite:
        resume_args = yaml.safe_load((Path(args.resume) / "config.yaml").read_text())
        vars(args).update(
            {k: v for k, v in vars(resume_args).items() if "resume" not in k}
        )

    args.ds_dir = Path(args.ds_dir)
    if args.ds_dir.is_dir():
        if args.resume:
            assert (args.ds_dir / "seeds_recorded.txt").exists()
        elif args.overwrite:
            shutil.rmtree(args.ds_dir)
        else:
            raise ValueError("There is already a dataset with this name")
    args.ds_dir.mkdir(exist_ok=True)

    print(type(args))
    (args.ds_dir / "config.yaml").write_text(yaml.dump(args))

    all_keys = record_dataset_local(
        ds_dir=args.ds_dir,
        scene_kwargs=args.scene_kwargs,
        start_seed=0,
        n_chunks=int(args.n_chunks),
        n_frames_per_chunk=int(args.n_frames_per_chunk),
        resume=args.resume,
        n_workers=args.n_workers,
    )

    n_train = int(args.train_ratio * len(all_keys))
    train_keys, val_keys = all_keys[:n_train], all_keys[n_train:]
    (args.ds_dir / "keys.pkl").write_bytes(pickle.dumps(all_keys))
    (args.ds_dir / "train_keys.pkl").write_bytes(pickle.dumps(train_keys))
    (args.ds_dir / "val_keys.pkl").write_bytes(pickle.dumps(val_keys))
