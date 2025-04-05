import matplotlib.pyplot as plt
from deps.cosypose.cosypose.datasets.datasets_cfg import make_scene_dataset

ds_name = "synthetic.ballvalve-1M.train"
scene_ds = make_scene_dataset(ds_name)
im, mask, obs = scene_ds[0]

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(im)
axarr[1].imshow(mask)

plt.show()
