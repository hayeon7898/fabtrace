# fabtrace

Wafer defect root-cause analysis and containment on a simulated fab (WM-811K).

## Data preparation

The original WM-811K file (`LSWMD.pkl`) is a Python 2 pickle that needs a lot of memory to load,
so it is converted once on Kaggle and only the compact output is used locally.

1. Open a Kaggle notebook, add the WM-811K dataset as input, and run `scripts/prepare_wm811k.py`
   (notebook: [kaggle-notebook](https://www.kaggle.com/code/haye0njung/wm-811k-preprocessing/edit)).
2. Download the outputs into `data/interim/`:
   - `wm811k_labeled_64.npz`: 172,950 labeled wafer maps, 64x64, values 0/1/2
   - `wm811k_labeled_64.parquet`: labels and original indices, same row order
3. Verify:

```bash
uv sync
uv run python scripts/check_data.py
```

| Class | Count |
| --- | --- |
| none | 147,431 |
| Edge-Ring | 9,680 |
| Edge-Loc | 5,189 |
| Center | 4,294 |
| Loc | 3,593 |
| Scratch | 1,193 |
| Random | 866 |
| Donut | 555 |
| Near-full | 149 |

`none` accounts for 85% of labeled maps, so macro-F1 is used as the main metric instead of accuracy.

![Sample wafer maps](docs/figures/wm811k_samples.png)