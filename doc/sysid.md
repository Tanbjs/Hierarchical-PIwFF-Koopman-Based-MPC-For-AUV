# System Identification — Details

*Paper §IV.* Three sequential scripts under [script/sysid/](../script/sysid/), backed by the [src/sysid/](../src/sysid/) library.

> **Reproducing the paper.** Running each stage **with no CLI flags** uses the paper-prescribed configuration (Table 1, Remark 3, Table 2 caption) and reproduces the manuscript results. The flags shown below exist so you can sweep alternative settings for your own experiments — but to cite the paper numbers, keep the defaults.

```mermaid
flowchart LR
    classDef artifact fill:#fafafa,stroke:#9a9a9a,stroke-width:1px,color:#1a1a1a
    classDef script   fill:#1a1a1a,stroke:#1a1a1a,stroke-width:1.2px,color:#ffffff
    classDef lib      fill:#ffffff,stroke:#1a1a1a,stroke-width:1.2px,color:#1a1a1a,stroke-dasharray:4 3

    D0[("<b>data/sysid/dataset/</b><br/>released closed-loop logs<br/>(NWU)")]:::artifact
    S0["<b>preprocess.py</b><br/>NWU&rarr;NED · Hampel · MA · split"]:::script
    D1[("<b>data/sysid/split/</b><br/>train · test symlinks")]:::artifact

    P0[("<b>params/sysid/</b><br/>dmdc.yaml · edmdc.yaml")]:::artifact
    L1["<i>src/sysid/</i><br/>filters · observables<br/>DMDc · EDMDc · FittedModel"]:::lib
    S1["<b>fit.py</b><br/>OLS regression in lifted space"]:::script
    R1[("<b>result/sysid/trained_model/</b><br/>A.npy · B.npy · scalers<br/>columns.json · metadata.json")]:::artifact

    S2["<b>validate.py</b><br/>one-/p-step prediction<br/>vs nonlinear baseline"]:::script
    L3["<i>src/utils/</i><br/>Fossen f_dyn (RK4)"]:::lib
    R2[("<b>result/sysid/validation/</b><br/>RMSE tables · prediction PDFs")]:::artifact

    D0 --> S0 --> D1
    D1 --> S1
    P0 --> S1
    L1 -.-> S1
    S1 --> R1
    D1 --> S2
    R1 --> S2
    L1 -.-> S2
    L3 -.-> S2
    S2 --> R2
```

## Stage 1 — Preprocess (`script/sysid/preprocess.py`)

Reads the released CSVs under `data/sysid/dataset/` and runs four sequential stages, each writing its own intermediate folder so cleaner/smoother configurations can coexist side-by-side:

1. **dataset &rarr; raw** — inject Euler angles, then flip NWU axes to NED (`data/sysid/raw/`). All downstream code operates in NED.
2. **raw &rarr; cleaned** — vectorized Hampel outlier filter (`data/sysid/cleaned/.../hampel_window{w}_sigma{n}.csv`).
3. **cleaned &rarr; smooth** — centered moving-average filter (`data/sysid/smooth/.../ma_window{w}.csv`).
4. **smooth &rarr; split** — sequential `sklearn.train_test_split` (seed `42`, ratio `0.85 / 0.15` &rarr; **25 train / 5 test** runs from the 30-run dataset, matching Table 1 of the paper); output is a tree of relative symlinks at `data/sysid/split/{train,test}/...`, mirroring the `kmc` reference pipeline.

**Paper defaults (recommended — reproduces Table 1):** `--hampel-window 5 --hampel-sigma 3 --ma-window 5 --ratio 0.85 0.15 --seed 42`. The paper specifies the train/test split and the sampling period `Ts = 0.1 s` from Table 1; it only requires "a Hampel filter" and "a short moving-average window," so the filter widths (5/3/5) are the project's reference choice and are baked in as the script defaults.

```bash
# Paper recipe — no flags needed.
python script/sysid/preprocess.py

# Play with it — sweep filter widths or split ratio.
python script/sysid/preprocess.py --hampel-window 7 --hampel-sigma 3 --ma-window 9
python script/sysid/preprocess.py --ratio 0.80 0.20 --seed 123
python script/sysid/preprocess.py --skip-raw           # reuse existing data/sysid/raw/
```

## Stage 2 — Fit Koopman model (`script/sysid/fit.py`)

Consumes a YAML config from [params/sysid/](../params/sysid/) (`dmdc.yaml` or `edmdc.yaml`) and the `train` split:

1. Select state / input / output feature columns by name groups; reorder so the **outputs are the first block of the lifted state** (`C = [I 0]`).
2. Fit `StandardScaler` (or `MinMaxScaler`) on the train concatenation; stack one-step pairs `(X_k, U_k, X_{k+1})`.
3. **DMDc** — identity lifting; OLS regression yields a 6×6 `A` and 6×6 `B` in the original velocity coordinates.
4. **eDMDc** — `PolynomialObservable` (**degree 2, no bias, no interaction-only**, per paper Remark 3) lifts to a 27-dim basis (linear + quadratic monomials), capturing quadratic drag and Coriolis cross-coupling. OLS regression yields a 27×27 `A` and 27×6 `B`.
5. Persist `A.npy`, `B.npy`, the three scalers (`joblib`), `columns.json`, the full config snapshot, and metadata (sample count, conditioning) under `result/sysid/trained_model/<method>/`.

**Paper defaults (recommended — reproduces Table 2):** the shipped [params/sysid/dmdc.yaml](../params/sysid/dmdc.yaml) and [params/sysid/edmdc.yaml](../params/sysid/edmdc.yaml) already encode the paper configuration — `StandardScaler(with_mean=true, with_std=true)`, eDMDc polynomial `degree: 2, include_bias: false, interaction_only: false`, OLS regression. Run them as-is:

```bash
# Paper recipe — no edits to the YAMLs.
python script/sysid/fit.py --config params/sysid/dmdc.yaml
python script/sysid/fit.py --config params/sysid/edmdc.yaml
```

To play with the model, copy a YAML and edit the scaler (`minmax` / `none`) or, for eDMDc, the polynomial `degree` / `interaction_only` / `include_bias` knobs — those alternatives are **not** the paper configuration.

The raw dataset was collected with a cascade PID–PID controller tracking a **diagonal-transit + sinusoidal-weaving** reference, with zero-mean Gaussian perturbation (std `σ = 5.0`, Table 1) injected on top of the PID command to excite the cross-coupled sway–yaw dynamics. **Regression is OLS only** (no ridge/lasso) — matching the `kmc` reference and the paper.

## Stage 3 — Validate (`script/sysid/validate.py`)

Implements the manuscript's open-loop prediction protocol on the held-out `test` split:

- **One-step** and **p-step sliding-window** prediction for DMDc, eDMDc, and the **nonlinear baseline** (Fossen 6-DOF `f_dyn` integrated with RK4 from [src/utils/](../src/utils/)).
- Per-channel **RMSE** partitioned into the *maneuvering* (`t < 60 s`) and *hovering* (`t ≥ 60 s`) phases used in paper Table 2 (`tf = 60 s` from Table 1).
- Per-trajectory and ensemble-averaged metric tables (SVG) plus 3×2 prediction plots (PDF, IEEE Access style) under `result/sysid/validation/`.

**Paper defaults (recommended — reproduces Table 2):** prediction horizon `Np = 10` (paper §IV multi-step protocol), sampling period `dt = 0.1 s` (Table 1). Both are the script defaults:

```bash
# Paper recipe — no flags needed.
python script/sysid/validate.py

# Play with it — sweep the rollout horizon.
python script/sysid/validate.py --step 20              # longer horizon (not the paper number)
python script/sysid/validate.py --step 5
```

**Outcome:** eDMDc wins on open-loop multi-step prediction; DMDc wins on the compute/accuracy trade for closed-loop control, so DMDc is the deployment target.