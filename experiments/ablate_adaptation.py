"""Switch off one adaptation choice at a time and rebuild the submission (for public-leaderboard diagnosis).
Usage (repo root): python -m experiments.ablate_adaptation nopool|nodisp
  nopool: no pooled station-normalised model for S0 wind
  nodisp: no dispersion-ratio correction (S0 wind still uses the pooled model, with its own ratio)
"""
import sys
import adapt

variant = sys.argv[1]
if variant == "nopool":
    adapt.POOLED_CELLS = []
elif variant == "nodisp":
    adapt.DEADBAND = (0.0, 1e9)
else:
    sys.exit(__doc__)
adapt.main(f"submission_abl_{variant}.csv", ["lad_seas", "gbm", "nn6a"], weights="blend_weights.json")
