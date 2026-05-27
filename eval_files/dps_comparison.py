"""DPS (Diffusion Posterior Sampling) comparison between diffusion and flow-matching models.

For each model we:
  1. Draw an unconditional sample to use as a reference image.
  2. Apply a rectangular inpainting mask.
  3. Run DPS (posterior_sample) to reconstruct the masked image.

The final figure shows, per model:
  reference | masked measurement | DPS reconstruction(s)

Saves to checkpoints/dps_comparison.png.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from zea import init_device

init_device()

import numpy as np
import matplotlib.pyplot as plt
import keras

from zea.models.diffusion import DiffusionModel  # noqa: F401
from zea.models.flow_matching import FlowMatchingModel  # noqa: F401
from zea.ops.ultrasound import scan_convert
from zea.visualize import set_mpl_style

set_mpl_style()

# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS = 50          # DPS diffusion steps
N_POSTERIOR = 2       # posterior samples per measurement
OMEGA = 0.5           # DPS step-size weight
SEED_REF = 0          # seed for unconditional reference images
SEED_DPS = 42         # seed for DPS

# Cardiac scan-convert geometry (must match training data)
RHO_RANGE = (0.0, 0.15)
THETA_RANGE = (-np.pi / 6, np.pi / 6)

MODELS = {
    "diffusion": os.path.join(ROOT, "checkpoints", "diffusion.keras"),
    "flow_matching": os.path.join(ROOT, "checkpoints", "flow_matching.keras"),
}

CUSTOM_OBJECTS = {
    "DiffusionModel": DiffusionModel,
    "FlowMatchingModel": FlowMatchingModel,
}

OUTPUT_PATH = os.path.join(ROOT, "checkpoints", "dps_comparison.png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_inpainting_mask(shape, fraction=0.4, rng=None):
    """Return a boolean keep-mask (True = observed) with a centred rectangle removed.

    Args:
        shape: ``(H, W, C)`` image shape.
        fraction: Fraction of height/width to mask out (centred).
        rng: Optional numpy RNG for reproducibility.

    Returns:
        float32 mask of shape ``(1, H, W, C)`` — 1 = observed, 0 = masked.
    """
    H, W, C = shape
    mask = np.ones((H, W, C), dtype=np.float32)
    h_start = int(H * (0.5 - fraction / 2))
    h_end = int(H * (0.5 + fraction / 2))
    w_start = int(W * (0.5 - fraction / 2))
    w_end = int(W * (0.5 + fraction / 2))
    mask[h_start:h_end, w_start:w_end, :] = 0.0
    return mask[np.newaxis]  # (1, H, W, C)


def to_scan_convert(img_hwc):
    """Apply scan-conversion to a single (H, W, C) image; returns (H', W') array."""
    img_hw = np.squeeze(img_hwc)
    sc, _ = scan_convert(img_hw, rho_range=RHO_RANGE, theta_range=THETA_RANGE, fill_value=0.0)
    return sc


def show(ax, img, title="", vmin=0.0, vmax=1.0):
    """Display a scan-converted image on ax."""
    ax.imshow(to_scan_convert(img), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

# columns: reference | measurement (masked) | posterior sample 1 | posterior sample 2 | …
N_COLS = 2 + N_POSTERIOR  # reference + masked + N posterior
N_ROWS = len(MODELS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.2, N_ROWS * 2.4))
if N_ROWS == 1:
    axes = axes[np.newaxis, :]

col_labels = ["Reference", "Masked measurement"] + [f"DPS sample {i + 1}" for i in range(N_POSTERIOR)]

for row, (model_name, ckpt_path) in enumerate(MODELS.items()):
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}  ({ckpt_path})")
    print(f"{'=' * 60}")

    model = keras.models.load_model(ckpt_path, custom_objects=CUSTOM_OBJECTS)

    # ── 1. Unconditional reference sample ─────────────────────────────────
    print("  Generating unconditional reference sample …")
    ref_seed = keras.random.SeedGenerator(SEED_REF + row)
    ref = model.sample(n_samples=1, n_steps=N_STEPS, verbose=False, seed=ref_seed)
    ref_np = np.array(ref)[0]  # (H, W, C)

    # ── 2. Build masked measurement ───────────────────────────────────────
    input_shape = tuple(ref_np.shape)  # (H, W, C)
    mask = make_inpainting_mask(input_shape, fraction=0.4)  # (1, H, W, C)
    measurement = ref_np[np.newaxis] * mask  # (1, H, W, C)

    print(f"  Image shape: {input_shape}  |  Masked fraction: 40% (centre rectangle)")

    # ── 3. DPS posterior sampling ─────────────────────────────────────────
    print(f"  Running DPS: {N_STEPS} steps, {N_POSTERIOR} posterior samples, omega={OMEGA} …")
    dps_seed = keras.random.SeedGenerator(SEED_DPS + row)
    posteriors = model.posterior_sample(
        measurements=measurement,          # (1, H, W, C)
        n_samples=N_POSTERIOR,
        n_steps=N_STEPS,
        mask=mask,
        omega=OMEGA,
        seed=dps_seed,
        verbose=True,
    )
    # posteriors shape: (1, N_POSTERIOR, H, W, C)
    posteriors_np = np.array(posteriors)[0]  # (N_POSTERIOR, H, W, C)

    # ── 4. Plot ───────────────────────────────────────────────────────────
    # Row label
    axes[row, 0].set_ylabel(
        model_name.replace("_", " ").title(),
        fontsize=10,
        rotation=90,
        labelpad=6,
        va="center",
    )

    show(axes[row, 0], ref_np, title=col_labels[0] if row == 0 else "")
    show(axes[row, 1], measurement[0], title=col_labels[1] if row == 0 else "")
    for i in range(N_POSTERIOR):
        show(axes[row, 2 + i], posteriors_np[i], title=col_labels[2 + i] if row == 0 else "")

fig.suptitle(f"DPS Inpainting Comparison  (n_steps={N_STEPS}, ω={OMEGA})", fontsize=12)
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved → {OUTPUT_PATH}")
