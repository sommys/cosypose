from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import numpy as np
from deps.cosypose.cosypose.datasets.datasets_cfg import make_scene_dataset

ds_id = "synthetic.ballvalve-falltest.train"
ds_dir = Path("/home/sommys/Documents/datagen/deps/cosypose/local_data/synt_datasets")

scene_ds = make_scene_dataset(ds_id)


def show_sample(ds, idx=0):
    im, mask, _ = ds[idx]
    _, axarr = plt.subplots(2, 1)
    axarr[0].imshow(im)
    axarr[1].imshow(mask)
    plt.show()


def make_gif(ds, save_dir, duration=1500):
    images = []
    masks = []
    for im, mask, _ in ds:
        images.append(im)
        masks.append(mask)

    with imageio.get_writer(save_dir / "im.gif", mode="I", duration=duration) as writer:
        for image in images:
            writer.append_data(image)  # type: ignore

    with imageio.get_writer(
        save_dir / "mask.gif", mode="I", duration=duration
    ) as writer:
        for mask in masks:
            cmap = cm.get_cmap("viridis", mask.max() + 1)
            mask = cmap(mask)
            mask = (mask[:, :, :3] * 255).astype(np.uint8)
            writer.append_data(mask)  # type: ignore


# print(len(scene_ds))
# show_sample(scene_ds, 4)
make_gif(scene_ds, ds_dir / scene_ds.name)
