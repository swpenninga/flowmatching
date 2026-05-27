"""Pure tf.data dataloader for EchoNetLVH HDF5 dataset.

Replaces the zea/grain Dataloader to avoid grain-TensorFlow threading
deadlocks. The entire pipeline is tf.data-native, which manages
prefetching and thread pools safely alongside ``model.fit()``.

Pipeline
--------
HDF5 files → interleave (h5py reads) → individual frames
          → clip / resize / normalise → shuffle → batch → prefetch

Usage
-----
.. code-block:: python

    from dataloader import build_dataset

    train_ds = build_dataset("/path/to/train", image_size=(128, 128), batch_size=64)
    val_ds   = build_dataset("/path/to/val",   image_size=(128, 128), batch_size=64, shuffle=False)
"""

import glob
import os

import h5py
import numpy as np
import tensorflow as tf


HDF5_KEY = "data/image"


def _count_samples(file_list: list[str]) -> int:
    """Return total number of frames across all HDF5 files."""
    total = 0
    for path in file_list:
        with h5py.File(path, "r") as f:
            total += f[HDF5_KEY].shape[0]
    return total


def _find_hdf5_files(data_path: str) -> list[str]:
    """Return sorted list of HDF5 files under *data_path*."""
    if os.path.isfile(data_path):
        return [data_path]
    files = sorted(
        glob.glob(os.path.join(data_path, "*.hdf5"))
        + glob.glob(os.path.join(data_path, "*.h5"))
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {data_path!r}")
    return files


def _read_frames(path_tensor) -> np.ndarray:
    """Read all frames from one HDF5 file. Runs inside tf.py_function."""
    path = path_tensor.numpy().decode("utf-8")
    with h5py.File(path, "r") as f:
        data = f[HDF5_KEY][:]          # (N, H, W) uint8
    return data.astype(np.float32)     # return float for downstream tf ops


def build_dataset(
    data_path: str,
    image_size: tuple,
    batch_size: int,
    shuffle: bool = True,
    cycle_length: int = 8,
    shuffle_buffer: int = 2000,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Build an infinite, prefetched ``tf.data.Dataset`` for EchoNetLVH.

    Args:
        data_path: Path to a directory (or single file) of ``.hdf5`` files.
        image_size: ``(height, width)`` to resize every frame to.
        batch_size: Number of frames per batch.
        shuffle: Shuffle both the file list and the frame-level buffer.
        cycle_length: Number of HDF5 files to interleave concurrently.
        shuffle_buffer: Frame-level shuffle buffer size.
        drop_remainder: Drop the final incomplete batch each epoch.

    Returns:
        An infinite ``tf.data.Dataset`` yielding tensors of shape
        ``(batch_size, H, W, 1)`` with values in ``[0, 1]``.
    """
    file_list = _find_hdf5_files(data_path)
    h, w = image_size

    # ------------------------------------------------------------------
    # 1. File-level dataset — optionally shuffled each repetition
    # ------------------------------------------------------------------
    file_ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        file_ds = file_ds.shuffle(len(file_list), reshuffle_each_iteration=True)

    # ------------------------------------------------------------------
    # 2. Expand each file into its individual frames via interleave.
    #    h5py releases the GIL during file I/O, so parallel reads work.
    # ------------------------------------------------------------------
    def load_file(path):
        frames = tf.py_function(_read_frames, [path], tf.float32)
        # shape[0] = N frames (unknown at graph build time), shape[1:] = H, W
        frames.set_shape([None, None, None])
        return tf.data.Dataset.from_tensor_slices(frames)  # yields (H, W) tensors

    frame_ds = file_ds.interleave(
        load_file,
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,   # allow reordering for maximum throughput
    )

    # ------------------------------------------------------------------
    # 3. Per-frame preprocessing (all native tf ops — no Python calls)
    # ------------------------------------------------------------------
    def preprocess(frame):
        frame = tf.clip_by_value(frame, 0.0, 255.0)         # clip to source range
        frame = tf.expand_dims(frame, axis=-1)               # (H, W) → (H, W, 1)
        frame = tf.image.resize(frame, [h, w])               # → (h, w, 1)
        frame = frame / 255.0                                # normalise to [0, 1]
        return frame

    ds = frame_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # 4. Frame-level shuffle, batch, repeat, prefetch
    # ------------------------------------------------------------------
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
