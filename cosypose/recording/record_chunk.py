import numpy as np
import pickle
import importlib
from pathlib import Path
from PIL import Image
from io import BytesIO
import sys
import os

from deps.cosypose.cosypose.recording.bop_recording_scene import BopRecordingScene


class SuppressStdout:
    def __enter__(self):
        self._stdout_fd = sys.__stdout__.fileno()
        self._stderr_fd = sys.__stderr__.fileno()
        self._stdout_copy = os.dup(self._stdout_fd)
        self._stderr_copy = os.dup(self._stderr_fd)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, self._stdout_fd)
        os.dup2(self._devnull, self._stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._stdout_copy, self._stdout_fd)
        os.dup2(self._stderr_copy, self._stderr_fd)
        os.close(self._devnull)


def get_cls(cls_str):
    split = cls_str.split(".")
    mod_name = ".".join(split[:-1])
    cls_name = split[-1]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _serialize_im(im, **pil_kwargs):
    im = Image.fromarray(np.asarray(im))
    im_buf = BytesIO()
    im.save(im_buf, **pil_kwargs)
    im_buf = im_buf.getvalue()
    return im_buf


def _get_dic_buf(state, jpeg=True, jpeg_compression=100):
    if jpeg:
        pil_kwargs = dict(format="JPEG", quality=jpeg_compression)
    else:
        pil_kwargs = dict(format="PNG", quality=100)

    del state["camera"]["depth"]
    state["camera"]["rgb"] = _serialize_im(state["camera"]["rgb"], **pil_kwargs)
    state["camera"]["mask"] = _serialize_im(
        state["camera"]["mask"], format="PNG", quality=100
    )
    return pickle.dumps(state)


def write_chunk(state_list, seed, ds_dir):
    key_to_buf = dict()
    dumps_dir = Path(ds_dir) / "dumps"
    dumps_dir.mkdir(exist_ok=True)

    for n, state in enumerate(state_list):
        key = f"{seed}-{n}"
        key_to_buf[key] = _get_dic_buf(state)

    # Write on disk
    for key, buf in key_to_buf.items():
        (dumps_dir / key).with_suffix(".pkl").write_bytes(buf)
    keys = list(key_to_buf.keys())
    return keys


def record_chunk(ds_dir, scene_kwargs, seed, n_frames):
    ds_dir = Path(ds_dir)
    ds_dir.mkdir(exist_ok=True)

    scene_kwargs["seed"] = seed
    scene = BopRecordingScene(**scene_kwargs)
    with SuppressStdout():
        scene.connect(load=True)
    state_list = []
    for _ in range(n_frames):
        state = scene.make_new_scene()
        state_list.append(state)
    keys = write_chunk(state_list, seed, ds_dir)
    with SuppressStdout():
        scene.disconnect()
    del scene
    return keys, seed
