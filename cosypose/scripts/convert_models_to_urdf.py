from pathlib import Path
import argparse
import shutil
import pymeshlab
from tqdm import tqdm

from cosypose.config import LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_object_dataset
from cosypose.libmesh import obj_to_urdf


def convert_obj_dataset_to_urdfs(obj_ds_name):
    obj_dataset = make_object_dataset(obj_ds_name)
    urdf_dir = LOCAL_DATA_DIR / "urdfs" / obj_ds_name
    urdf_dir.mkdir(exist_ok=True, parents=True)
    for n in tqdm(range(len(obj_dataset))):
        obj = obj_dataset[n]
        ply_path = Path(obj["mesh_path"])
        out_dir = urdf_dir / obj["label"]
        out_dir.mkdir(exist_ok=True)
        obj_path = ply_path.with_suffix(".obj")
        out_obj_path = out_dir / obj_path.name
        if obj_path.exists():
            shutil.copy(obj_path, out_obj_path)
        else:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(ply_path.as_posix())
            ms.save_current_mesh(out_obj_path.as_posix())
        urdf_path = out_obj_path.with_suffix(".urdf")
        obj_to_urdf(out_obj_path, urdf_path)


def main():
    parser = argparse.ArgumentParser("3D ply object models -> pybullet URDF converter")
    parser.add_argument("--models", default="", type=str)
    args = parser.parse_args()
    convert_obj_dataset_to_urdfs(args.models)


if __name__ == "__main__":
    main()
