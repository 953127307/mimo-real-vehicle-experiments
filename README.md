# Real-Vehicle Experiments

## Files

- `generate_experimental_results.py` — Processing entry: verifies the data against the manifest (SHA-256), computes the Table VI metrics, and draws Fig. 4.
- `data/` — Frozen inputs: campaign manifest, K504 training dataset, validated weights, 20 closed-loop traces (`traces/`), platform photo.
- `runs/` — Per-run outputs (JSON and figures); `latest_experiments.txt` points to the newest run.
- `firmware/` — Copy of the flashed firmware v1.7.4 source.
- `gui/` — Vehicle-side experiment GUI source and EXE.

## Usage

```bash
python generate_experimental_results.py                  # creates a new run folder and draws the figure; the figure is copied to ../submission/figures/
python generate_experimental_results.py --run <folder>   # rewrite an existing run
cat latest_experiments.txt                               # newest run name

# Vehicle experiment GUI:
gui/dist/MIMO-Joystick-Paper.exe                         # or: python gui/joystick_car.py

# Rebuild the EXE after modifying the GUI source:
python -m PyInstaller gui/MIMO-Joystick-Paper.spec --noconfirm
```
