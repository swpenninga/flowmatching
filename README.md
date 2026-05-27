# Flowmatching
Training and comparison of flowmatching w.r.t. DDIM diffusion.

Done on EchonetLVH dataset using zea implementations

- `dataloader.py` contains a simple tensorflow loader
- `train.py` trains flowmatching/diffusion model and stores to `checkpoints` folder.
- In `eval_files` folder, for after training:
  - `plot_val_nv_loss.py` plots training statistics of both models.
  - `sample.py` plots some samples vs num_steps for both models.
  - `dps_comparison.py` gives a comparison for DPS inpainting for a single evaluation.
  - `dps_scanlines.ipynb` loops over some samples and reports the DPS loss of both models.
