# Model artifacts

`artifacts/models/` contains the 12 fitted estimators used for the published
frozen holdout: four estimators for each of the 60, 300, and 600 second horizons.

The two-stage strategy uses both the fill classifier and adverse-selection
regressor. The unified strategies use the short- or long-context regressor. Model
metadata records training dates, row counts, feature counts, label version, and
training iterations.

SHA-256 checksums are stored in `artifacts/models/manifest.json`.

Joblib files use Python pickle semantics. They should be loaded only from a trusted
repository revision. They may require the scikit-learn version recorded in the
manifest; retraining from source is the most portable option.
