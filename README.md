# Blind Forecast: Five-Day Weather Forecasting Under Distribution Shift

Individual entry to the Amalthea x SCALE *Blind Forecast* competition (Kaggle competition `blind-forecast`). The original task description is reproduced in [docs/competition_description.md](docs/competition_description.md).

## 1. Summary

The task is to forecast daily maximum temperature, minimum temperature, relative humidity and wind speed for the five days following a ten-day observation window, at ten anonymous weather stations, without dates, coordinates or place names. The score is the mean, over the four variables, of the mean absolute error divided by the mean absolute error of a station-and-day-of-year climatology (1.000 corresponds to climatology; lower is better).

The final system combines three components:

1. a representation of every series as scaled anomalies from a per-station harmonic climatology;
2. a blend of three forecasters trained on those anomalies (per-station least-absolute-deviation regression, gradient-boosted trees, and a pooled neural network trained with shift and scale augmentation);
3. a test-time adaptation step that corrects for the fact that the test windows are drawn from a regime that differs systematically from the training period. This step produced the largest single improvement.

No external data of any kind is used. Only the four files supplied by the organisers are read.

| Stage | Public leaderboard (30% of windows) |
|---|---|
| Per-station LAD regression with season interactions | 0.7364 |
| + adaptation of the climatology to the test period | 0.6884 |
| + dispersion correction and pooled model for one station-variable | 0.6869 |
| + neural network in the blend | 0.6797 |
| + six seeds and shift/scale augmentation | **0.6753** |

For reference, the organisers' benchmarks are 1.000 (climatology), 0.822 (persistence) and 0.768 (the SCALE benchmark). The public leaderboard covers 30% of the windows and is noisy; differences of about 0.002 or less between submissions should not be interpreted.

## 2. Data

| File | Content |
|---|---|
| `train.csv` | 10 stations (`S0` to `S9`), 27,394 consecutive daily records each (about 75 years; 273,940 station-days in total). Columns: `station`, `day_index`, `doy`, `tmax`, `tmin`, `rh`, `wind`. There are no gaps or missing values. |
| `test.csv` | 890 windows (89 per station). Each has 10 observed days (`step` -9 to 0) and 5 unobserved days (`step` 1 to 5, values missing). The `regime` column (A to D) is a season code that is determined by `doy`. |
| `scales.csv` | One divisor per variable (tmax 2.195, tmin 1.933, rh 9.415, wind 3.315): the climatology MAE of that variable. |
| `sample_submission.csv` | 17,800 rows (890 windows x 5 steps x 4 variables); `value` is the prediction divided by the variable's scale. |

These files are not included in this repository and must be downloaded from the competition page.

## 3. Method

### 3.1 Anomaly representation

For every station and variable, a climatology is fitted by ordinary least squares on a constant and four sine/cosine harmonic pairs of the day of year (`doy / 365.25`), giving nine coefficients per station and variable. Observations are converted to scaled anomalies

    a = (x - climatology(doy)) / scale

with the per-variable `scale` from `scales.csv`. All models predict the scaled anomalies of the five future days. When a submission is written, the climatology is added back and the result is divided by the scale. Scaling by the climatology error makes the four variables commensurate, so that a plain mean absolute error on scaled anomalies is the competition metric up to the choice of climatology. Every loss used is an L1 loss, which matches the metric.

Windows are all stride-one sequences of 10 input days and 5 target days within a station (about 273,800 windows). Windows never straddle a validation boundary.

### 3.2 Forecasters

**Per-station LAD regression** (`linear.py`, model name `lad_seas`). The design matrix contains the 40 lagged anomalies (10 days x 4 variables), a constant and first and second seasonal harmonics of the last observed day, and the products of the 40 lags with the first-harmonic sine and cosine, which lets the dynamics vary with the season (125 features). Twenty outputs (5 steps x 4 variables) are fitted per station by iteratively reweighted least squares for the least-absolute-deviation criterion (8 iterations, every third window, ridge initialisation).

**Gradient-boosted trees** (`gbm.py`, `gbm`). One pooled LightGBM model for each of the 20 (step, variable) targets with an L1 objective (300 trees, learning rate 0.05, 31 leaves, minimum 100 samples per leaf, row and column subsampling 0.7, L2 penalty 5). Features: the 40 lags, short-term tendencies, means over the last 3 and 5 days and the whole window, window standard deviation, the difference between the means of the last and first five days, the recent tmax-tmin range, mean absolute anomaly, seasonal harmonics, and the station as a categorical variable.

**Neural network** (`nn.py`, `nn6a`). A residual multilayer perceptron: the 40 lags and four seasonal features, concatenated with an 8-dimensional station embedding, pass through a linear skip connection plus three hidden layers of 256 GELU units, producing all 20 outputs jointly. Training uses the L1 loss, AdamW (learning rate 0.002, weight decay 1e-4, one-cycle schedule), batch size 1024 and 14 epochs on all stride-one windows. Six independently seeded networks are averaged.

*Shift and scale augmentation.* For a random half of each mini-batch, inputs and targets are transformed together as `x' = c * x + o`, with a per-window, per-variable offset `o ~ N(0, 0.3^2)` and scale `c = exp(N(0, 0.2^2))`. This teaches the network to be equivariant to the level and variability shifts described in section 3.4 and costs nothing on unperturbed data (section 5).

### 3.3 Blending

For each of the 20 (step, variable) cells, non-negative blend weights summing to one are chosen on a grid of resolution 0.1 to minimise the mean absolute error on out-of-fold predictions, followed by a scalar shrink factor chosen from {0.90, 0.95, 1.00, 1.05, 1.10, 1.15}. The weights are fitted on out-of-fold predictions of five time-blocked folds (section 4). Averaged over cells, the network receives about 80% of the weight, LightGBM about 15% and the linear model about 5%.

### 3.4 Test-time adaptation to a shifted regime

Inspection of the test inputs shows that the test windows do not come from the training period. No test window appears verbatim in the training series, and the test anomalies relative to the training climatology are far from zero. For example, at station `S8` the mean tmax anomaly is +1.8 scaled units (about 4 degrees C), tmin +3.4 (about 6.6 degrees C) and relative humidity -2.0 (about 18 points), with an amplitude that depends on the season. By contrast, decadal wander in the training anomalies is about 0.3 scaled units. A model trained on zero-mean anomalies interprets such a persistent offset as transient weather and lets it decay, which systematically under-predicts the offset.

The correction uses only the unlabelled test inputs (the ten observed days of each window), which the competition files provide, and proceeds in three steps:

1. **Seasonal offset.** For every (station, variable) cell, the test-input anomalies are regressed on the first `k` harmonics of the day of year (ridge-regularised; 89 windows, or 890 observed days, per station). The fitted offset is added to the climatology of that cell. The harmonic order `k` in {0, 1, 2} is chosen per cell by repeated 50/50 splits of the test windows (20 repetitions) using held-out absolute residuals; `k = 1` is retained unless another order reduces the held-out error by more than 2%.
2. **Dispersion correction.** For each cell, the ratio of the mean absolute residual of the test inputs (after step 1) to the mean absolute anomaly of the training series on the same days of the year (matched within +/- 10 days) is computed. Where the ratio lies outside [0.8, 1.25], inputs are divided by it before prediction and the predicted anomalies are multiplied by it afterwards. Ratios are clipped to [0.5, 4]. Large ratios occur for wind at `S0`, `S6` and `S8` (2.9, 1.8, 2.7), for tmax at `S0` and `S9` (1.35, 1.45), for tmin at `S9` (1.25) and for humidity at `S8` (1.26).
3. **Pooled model for one cell.** In training, wind at `S0` is almost constant noise (lag-1 autocorrelation 0.25, against 0.49 in the test inputs and about 0.5 to 0.6 at the other stations), so its own history gives unrepresentative dynamics. The forecast for that single cell comes from a LAD regression trained on the station-normalised anomalies of the seven stations whose train and test behaviour agree (`S1` to `S5`, `S7`, `S9`). A leave-one-station-out experiment shows that pooling in this way costs only about 1.5% of the wind MAE at a normal station.

## 4. Validation

Cross-validation uses five contiguous time blocks of about 15 years. For each fold, the climatology is refitted on the training part, models are trained on windows lying wholly outside the validation block, and the validation block is scored on every window it contains. Random window splits are not used because adjacent windows overlap by up to nine days. Scores on this validation are not on the leaderboard scale (the local climatology scores 0.824, not 1.000, and persistence 0.774) and are used only to compare models.

Cross-validation cannot detect the distribution shift of section 3.4, because the training series contains no comparable shift. Robustness to the shift was therefore examined by simulation: the estimated test offsets and dispersion ratios were injected into validation windows and the complete adaptation pipeline was run with the per-station ridge model. In that experiment the estimated correction reached a mean error of 0.6798, equal to that of an oracle that knows the injected shift exactly, whereas applying no correction gave 0.720.

## 5. Results

### 5.1 Cross-validation

| Model | Five-fold CV score |
|---|---|
| Climatology | 0.824 |
| Persistence | 0.774 |
| Per-station ridge, season interactions | 0.641 |
| Per-station LAD, season interactions (`lad_seas`) | 0.641 |
| LightGBM (`gbm`) | 0.637 |
| Neural network, 2 seeds | 0.627 |
| Neural network, 6 seeds | 0.626 |
| Neural network, 6 seeds, augmentation (`nn6a`) | 0.625 |
| Blend `lad_seas` + `gbm` + `nn6a` (cross-fitted weights) | **0.625** |
| Analog forecaster (40 nearest trajectories, median) | 0.682 |

Skill is concentrated at short horizons: for the neural network, the per-step score rises from about 0.44 at day 1 to about 0.72 at day 5. Wind is the least predictable variable (about 0.70) and temperature the most (about 0.59).

### 5.2 Ablations and negative results

| Experiment | Outcome |
|---|---|
| Shift/scale augmentation, one seed (folds 0, 2, 4) | clean 0.6299 to 0.6275; mildly perturbed validation 0.668 to 0.651; strongly perturbed 0.79 to 0.73 |
| GRU in place of the MLP, same augmentation | worse in every fold (0.6302 against 0.6275) |
| Wider and longer MLP (512 units, 24 epochs) | worse (0.6477): overfits |
| Smaller or shorter MLP (128 units / 10 epochs; 8 epochs) | worse (0.6336; 0.6319) |
| Station-normalised network inputs | no change (0.6305 against 0.6298) |
| More climatology harmonics (4 to 8) | ridge CV 0.6408 to 0.6403: negligible |
| Trend term in the climatology | not supported: no trend in training anomalies |
| Clipping forecasts to physical bounds | no forecast violates humidity <= 100, wind >= 0 or tmin <= tmax |
| Offset harmonic order 0 / 1 / 2 / 3 (held-out residual, mean over cells) | 0.890 / 0.877 / 0.872 / 0.887; order 2 helps specific cells (for example `S8` tmin, 1.909 to 1.758) |
| Analog forecaster in the blend | receives about 2% weight; dropped |
| Generative (GAN) synthetic data; station-specific modelling using the identity of the IIT Gandhinagar station | not pursued: the former adds no information beyond the 274,000 station-days available, and the latter would require knowledge from outside the provided files |

## 6. Reproducibility

### 6.1 Environment

Python 3.12 with `numpy` 1.26, `pandas` 2.2, `lightgbm` 4.7 and `torch` 2.9 (see `requirements.txt`). The four competition files `train.csv`, `test.csv`, `scales.csv` and `sample_submission.csv` must be present in the working directory.

### 6.2 Running

    pip install -r requirements.txt
    python reproduce.py --out submission.csv

`reproduce.py` performs the complete final procedure: climatology, test-time adaptation, fitting of the three forecasters on all of `train.csv`, blending and writing of the file, followed by an automatic check that the identifiers match `sample_submission.csv` and that there are no missing values. The neural network uses a CUDA GPU when one is available and the CPU otherwise.

The blend weights are read from `blend_weights.json`. That file was produced by this repository's code (`python blend.py export blend_weights.json lad_seas gbm nn6a`) from the five-fold out-of-fold predictions of the same three models on `train.csv`; regenerating it requires retraining every model five times and exceeds the 30-minute budget, which is why it is stored. `python reproduce.py --refit-weights` regenerates it from scratch for verification.

Options: `--no-ksel` uses a first-harmonic offset in every cell (the configuration that scored 0.6753 on the public leaderboard), and `--compare other.csv` reports the mean absolute difference from an existing submission.

### 6.3 Runtime and determinism

On an 8-thread CPU (no GPU) the complete default run takes about 4.7 minutes. On a CPU the output is deterministic: a repeated run reproduced the stored per-cell-offset submission with a mean absolute difference of 0.00000 scaled units. A GPU run was not tested; GPU kernels and random streams differ, so small numerical differences in the network's predictions are to be expected.

## 7. Compliance with the competition rules

* **Data.** Only the four supplied files are used. No reanalysis, station record, forecast, API or other external weather information enters the pipeline, and no guess of the stations' identities is used.
* **Use of the test inputs.** The adaptation of section 3.4 estimates offsets, dispersion ratios and a model-selection choice from the unlabelled observed days of the test windows. No test targets are available or used. This transductive use of the provided test inputs is disclosed here because it is a methodological choice that other entrants may treat differently.
* **Public libraries.** Only public libraries are used; no pretrained models of any kind.
* **Private leaderboard.** No attempt was made to infer private-leaderboard scores. Model selection relied on local cross-validation, simulation and, secondarily, the public score.

## 8. Limitations

* The adaptation parameters are estimated from 89 windows per station and could be imperfect in cells where the shift or the variability is large and strongly seasonal (for example humidity and wind at `S8`).
* The choice of which cells receive a dispersion correction and a pooled model was made by inspecting the test inputs, so it cannot be validated by cross-validation; it was supported by simulation and by the public score, which covers 30% of the windows.
* The harmonic-order selection and its 2% margin are heuristics. Their expected effect (about 0.001 to 0.002) is below the resolution of the public leaderboard.

## 9. Repository layout

| Path | Purpose |
|---|---|
| `reproduce.py` | End-to-end reproduction of the final submission |
| `blend_weights.json` | Stored blend weights and shrink factors (generated by `blend.py export`) |
| `common.py` | Data loading, climatology, anomalies, windowing, cross-validation folds, test loading, submission writer with validity checks |
| `linear.py`, `gbm.py`, `nn.py`, `analog.py` | The forecasters (the analog forecaster and the GRU class in `nn.py` are retained as documented negative results) |
| `oof.py` | Computes out-of-fold and test predictions for a model and caches them in `oof/` |
| `blend.py` | Blend-weight fitting, cross-fitted blend evaluation, weight export |
| `adapt.py` | Test-time adaptation and construction of the final blended submission |
| `experiments/` | The individual studies behind sections 3 to 5 (run from the repository root with `python -m experiments.<name>`) |
| `docs/competition_description.md` | The organisers' task description |
