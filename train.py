"""Training script for comparing DiffusionModel and FlowMatchingModel on EchoNetLVH.

Both models use an identical time-conditional UNet architecture and are trained
with the same zea Dataloader on the EchoNetLVH dataset so that results are
directly comparable.

Usage
-----
Train diffusion model::

    python train.py --model diffusion --data_path /path/to/echonetlvh/train

Train flow-matching model::

    python train.py --model flow_matching --data_path /path/to/echonetlvh/train

All hyper-parameters are exposed as CLI flags; see ``python train.py --help``.
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np

import keras  # noqa: E402

from zea import log, init_device  # noqa: E402
from zea.models.diffusion import DiffusionModel  # noqa: E402
from zea.models.flow_matching import FlowMatchingModel  # noqa: E402

from dataloader import build_dataset, _count_samples, _find_hdf5_files  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
TRAIN_DATA_PATH = "/data/USBMD_datasets/EchoNet-LVH/train"
VAL_DATA_PATH = "/data/USBMD_datasets/EchoNet-LVH/val"

# ---------------------------------------------------------------------------
# Shared UNet architecture
# ---------------------------------------------------------------------------
# Both models use get_time_conditional_unetwork with these exact kwargs so that
# the number of trainable parameters is identical.
UNET_KWARGS = dict(
    widths=[64, 96, 128, 256],
    block_depth=3,
    embedding_dims=64,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    normalization="group_norm",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DiffusionModel or FlowMatchingModel on EchoNetLVH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- model choice ---------------------------------------------------
    parser.add_argument(
        "--model",
        choices=["diffusion", "flow_matching"],
        required=True,
        help="Which generative model to train.",
    )

    # ---- data -----------------------------------------------------------
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Spatial size to resize all frames to.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    # ---- training -------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--ema_val",
        type=float,
        default=0.999,
        help="EMA coefficient for the inference network.",
    )

    # ---- output ---------------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory in which to save the trained model.",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        default=True,
        help="Only save the checkpoint that achieves the best validation loss.",
    )

    return parser.parse_args()


class GenerationPlotCallback(keras.callbacks.Callback):
    """Plot unconditionally generated images at several sampling-step counts.

    At the end of every ``plot_every`` epochs the EMA network is used to
    generate ``n_samples`` images with 10, 20, 50 and 100 reverse steps.
    Each step-count gets one row in the figure; each column is one sample.

    Figures are saved to ``<output_dir>/samples/epoch_{epoch:04d}.png``.

    Args:
        output_dir: Root directory; plots go into ``<output_dir>/samples/``.
        n_samples: Number of images per row (per step count).
        step_counts: Sequence of reverse-diffusion step counts to compare.
        plot_every: Plot once every this many epochs (default 1).
    """

    def __init__(self, output_dir, n_samples=8, step_counts=(10, 20, 50, 100), plot_every=1):
        super().__init__()
        self.samples_dir = os.path.join(output_dir, "samples")
        self.n_samples = n_samples
        self.step_counts = list(step_counts)
        self.plot_every = plot_every
        os.makedirs(self.samples_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_every != 0:
            return

        n_rows = len(self.step_counts)
        fig, axes = plt.subplots(
            n_rows,
            self.n_samples,
            figsize=(self.n_samples * 2, n_rows * 2),
        )
        # Ensure axes is always 2-D
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, n_steps in enumerate(self.step_counts):
            samples = self.model.sample(n_samples=self.n_samples, n_steps=n_steps)
            samples_np = np.array(samples)  # (N, H, W, C)

            for col in range(self.n_samples):
                ax = axes[row, col]
                img = np.squeeze(samples_np[col])  # (H, W)
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"{n_steps} steps", fontsize=9)

        fig.suptitle(f"Epoch {epoch + 1}", fontsize=11)
        fig.tight_layout()

        save_path = os.path.join(self.samples_dir, f"epoch_{epoch + 1:04d}.png")
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved sample plot → {save_path}")


def build_model(model_type, input_shape, ema_val):
    """Instantiate the requested model with a shared UNet architecture.

    Args:
        model_type: ``"diffusion"`` or ``"flow_matching"``.
        input_shape: ``(height, width, channels)`` tuple.
        ema_val: EMA coefficient.

    Returns:
        Compiled Keras model (either :class:`DiffusionModel` or
        :class:`FlowMatchingModel`).
    """
    common_kwargs = dict(
        input_shape=input_shape,
        input_range=(-1, 1),
        network_name="unet_time_conditional",
        network_kwargs=UNET_KWARGS,
        ema_val=ema_val,
        guidance=None,  # training only – no operator / guidance needed
        operator=None,
    )

    if model_type == "diffusion":
        model = DiffusionModel(
            **common_kwargs,
            min_signal_rate=0.02,
            max_signal_rate=0.95,
            name="diffusion_model",
        )
    else:
        model = FlowMatchingModel(
            **common_kwargs,
            name="flow_matching_model",
        )

    return model


def main():
    args = parse_args()

    init_device()
    log.info(f"Backend : {keras.backend}")
    log.info(f"Model   : {args.model}")
    log.info(f"Train data : {TRAIN_DATA_PATH}")
    log.info(f"Val data   : {VAL_DATA_PATH}")

    image_size = tuple(args.image_size)
    input_shape = (*image_size, 1)

    # ---- dataloaders ----------------------------------------------------
    train_ds = build_dataset(
        TRAIN_DATA_PATH,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    log.info(f"Train files   : {len(_find_hdf5_files(TRAIN_DATA_PATH))}")
    log.info(f"Train samples : {_count_samples(_find_hdf5_files(TRAIN_DATA_PATH))}")

    val_ds = build_dataset(
        VAL_DATA_PATH,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    log.info(f"Val samples   : {_count_samples(_find_hdf5_files(VAL_DATA_PATH))}")

    # ---- model ----------------------------------------------------------
    model = build_model(args.model, input_shape, args.ema_val)
    steps_per_epoch = 167965 // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6 / args.lr,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
        loss=keras.losses.MeanSquaredError(),
    )
    log.info(f"UNet trainable parameters : {model.network.count_params():,}")

    # ---- callbacks ------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{args.model}.keras")

    monitor_metric = "val_v_loss" if args.model == "flow_matching" else "val_n_loss"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=args.save_best_only,
            monitor=monitor_metric,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(args.output_dir, f"{args.model}_training_log.csv"),
            append=True,
        ),
        GenerationPlotCallback(
            output_dir=os.path.join(args.output_dir, args.model),
            n_samples=4,
            step_counts=(5, 10, 20, 50),
            plot_every=1,
        ),
    ]

    dummy_images = np.zeros((1, *input_shape), dtype="float32")
    dummy_noise_vars = np.zeros((1, 1, 1, 1), dtype="float32")
    model([dummy_images, dummy_noise_vars])

    # ---- training -------------------------------------------------------
    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, # 10% of dataset length
        validation_data=val_ds,
        validation_steps=18694//args.batch_size, # 10% of dataset length
        callbacks=callbacks,
    )

    final_path = os.path.join(args.output_dir, f"{args.model}_final.keras")
    model.save(final_path)
    log.info(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
