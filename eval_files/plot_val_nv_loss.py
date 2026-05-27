import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from zea.visualize import set_mpl_style
set_mpl_style()

diffusion = pd.read_csv(os.path.join(ROOT, "checkpoints", "diffusion_training_log.csv"))
flow = pd.read_csv(os.path.join(ROOT, "checkpoints", "flow_matching_training_log.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# val_i_loss: both models
axes[0].plot(diffusion["epoch"], diffusion["val_i_loss"], label="Diffusion")
axes[0].plot(flow["epoch"], flow["val_i_loss"], label="Flow Matching")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Validation Image Loss")
axes[0].legend()

# val_n_loss and val_v_loss overlaid
axes[1].plot(diffusion["epoch"], diffusion["val_n_loss"], label="Diffusion (val_n_loss)")
axes[1].plot(flow["epoch"], flow["val_v_loss"], label="Flow Matching (val_v_loss)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Validation Noise / Velocity Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "checkpoints", "val_losses.png"), dpi=150)
plt.show()

# val_i_loss zoomed in
fig2, ax = plt.subplots(figsize=(6, 5))
ax.plot(diffusion["epoch"], diffusion["val_i_loss"], label="Diffusion")
ax.plot(flow["epoch"], flow["val_i_loss"], label="Flow Matching")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Validation Image Loss (zoomed)")
ax.set_ylim(0.008, 0.015)
ax.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, "checkpoints", "val_i_loss_zoomed.png"), dpi=150)
plt.show()
