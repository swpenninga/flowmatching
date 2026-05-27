"""Unconditional sampling from both trained models at multiple step counts.

Generates a grid for each model: rows = step counts, columns = samples.
Saves to checkpoints/diffusion/samples/ and checkpoints/flow_matching/samples/.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from zea import init_device
init_device()

import time
import numpy as np
import matplotlib.pyplot as plt
import keras

from zea.models.diffusion import DiffusionModel  # noqa: F401
from zea.models.flow_matching import FlowMatchingModel  # noqa: F401
from zea.ops.ultrasound import scan_convert
from zea.visualize import set_mpl_style

set_mpl_style()

STEP_COUNTS = [15, 20, 50, 100]
N_SAMPLES = 2

# Cardiac phased-array sector geometry
RHO_RANGE = (0.0, 0.15)       # depth: 0–15 cm
THETA_RANGE = (-np.pi / 6, np.pi / 6)  # azimuth: ±45°

MODELS = {
    "diffusion": os.path.join(ROOT, "checkpoints", "diffusion.keras"),
    "flow_matching": os.path.join(ROOT, "checkpoints", "flow_matching.keras"),
}


def plot_samples(model, model_name, step_counts, n_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_rows = len(step_counts)

    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # warmup call to absorb TF graph tracing before timed runs
    print(f"  [{model_name}] warmup...")
    model.sample(n_samples=1, n_steps=step_counts[0], verbose=False, seed=keras.random.SeedGenerator(99))

    for row, n_steps in enumerate(step_counts):
        print(f"  [{model_name}] sampling with {n_steps} steps...")
        t0 = time.perf_counter()
        samples = model.sample(n_samples=n_samples, n_steps=n_steps, verbose=False, seed=keras.random.SeedGenerator(row))
        elapsed = time.perf_counter() - t0
        samples_np = np.array(samples)

        for col in range(n_samples):
            ax = axes[row, col]
            img = np.squeeze(samples_np[col])  # (H, W)
            img_sc, _ = scan_convert(
                img,
                rho_range=RHO_RANGE,
                theta_range=THETA_RANGE,
                fill_value=0.0,
            )
            ax.imshow(img_sc, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{n_steps} steps ({elapsed:.2f}s)", fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(model_name.replace("_", " ").title(), fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(output_dir, "unconditional_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


CUSTOM_OBJECTS = {
    "DiffusionModel": DiffusionModel,
    "FlowMatchingModel": FlowMatchingModel,
}

for model_name, checkpoint_path in MODELS.items():
    print(f"\nLoading {model_name} from {checkpoint_path} ...")
    model = keras.models.load_model(checkpoint_path, custom_objects=CUSTOM_OBJECTS)
    output_dir = os.path.join(ROOT, "checkpoints", model_name, "samples")
    plot_samples(model, model_name, STEP_COUNTS, N_SAMPLES, output_dir)
