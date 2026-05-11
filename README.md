# M.Stat Resampling Project: Bootstrap Variance for G-Computation

**Project Author:** Swagato Das (Roll No: MB2534, Indian Statistical Institute, Kolkata)  
**Original Paper Authors:** Paul N. Zivich, Haidong Lu 

---

## 🎯 Project Objective

This repository is an extension and methodological pivot of the paper *"Constructing g-computation estimators: two case studies in selection bias"* by Zivich and Lu. 

While the original authors effectively solve treatment-induced selection bias and complex confounding using **Modified** and **Iterated** g-computation, they conduct statistical inference using an analytically complex M-estimation framework (empirical sandwich variance). 

**The primary goal of this M.Stat project is to replicate their causal findings while entirely replacing the M-estimation framework with rigorous resampling methods.** By breaking down their stacked estimating equations into sequential Generalized Linear Models (GLMs), this project successfully implements and scales:
1. **Non-parametric Bootstrap**
2. **$m$-out-of-$n$ Bootstrap** (scaled by $\sqrt{m/n}$)
3. **Delete-$d$ Jackknife** (scaled by $\sqrt{(n-d)/d}$)

### Estimators Evaluated
* **Complete-case (Naïve):** Ignores unmeasured selection bias.
* **Standard g-computation:** Fails under treatment-induced selection bias due to topological restrictions.
* **Modified g-computation (Case 1):** Correctly intervenes on the treatment exclusively within the selected stratum.
* **Iterated g-computation (Case 2):** Correctly utilizes nested expectations to bridge spatially separated confounders.
* **Inverse Probability Weighting (IPW):** Included as the classical standard for comparison.

---

## 🚀 Repository Execution Guide

### Prerequisites
Ensure you have Python 3.9+ installed. The project requires the original dependencies plus a few additions for the bootstrap pipelines:
```bash
pip install numpy scipy pandas statsmodels tqdm delicatessen==3.0

```

### Automated Execution (Recommended)

You can run the entire project pipeline using the provided shell script. This will sequentially output the single-dataset variance tables (Tables 1 & 2) and the heavy Monte Carlo simulations (Tables 3 & 4) used in the report.

```bash
chmod +x run_simulations.sh
./run_simulations.sh

```

### Manual Execution Mapping

If you prefer to run the bootstrap scripts individually:

* `python code_2cases.py` $\rightarrow$ Generates Single-Dataset Variance Scaling (Tables 1 & 2).
* `python simulations/sim1_boot.py` $\rightarrow$ Generates Frequentist Properties for Case 1 (Table 3).
* `python simulations/sim2_boot.py` $\rightarrow$ Generates Frequentist Properties for Case 2 (Table 4).

*(Note: The Monte Carlo scripts contain nested computational loops. Case 2 may take roughly ~35 minutes to execute on standard hardware).*

---

## 📁 File Manifesto

### New Project Files (Bootstrap Pipelines)

* `run_simulations.sh`: Master shell script to execute the M.Stat resampling pipelines.
* `code_2cases.py`: Applies sequential GLMs and all three resampling algorithms to single datasets.
* `simulations/sim1_boot.py`: Monte Carlo experiment nesting the bootstrap for Case 1.
* `simulations/sim2_boot.py`: Monte Carlo experiment nesting the bootstrap for Case 2.

### Original Paper Assets

`data/`

* `example1.csv`: Single simulated data set from the first case study.
* `example2.csv`: Single simulated data set from the second case study.

`simulations/` *(Original M-Estimation Code)*

* `dgm.py`: Data generating mechanisms used across all simulations.
* `estfun.py`: Original estimating functions for `delicatessen`.
* `postprocess.py`: Simulation result processing helper functions.
* `sim_case-study1.py`: Original M-estimation experiment for Case 1.
* `sim_case-study2.py`: Original M-estimation experiment for Case 2.

`examples.ipynb` & `examples.Rmd`

* Python and R markdown notebooks walking through the baseline M-estimation estimators applied to the `data/` folder.

---

## 💻 System Details (Original Environment)

**Python: 3.9.4**

* Dependencies: `NumPy` (1.25.2), `SciPy` (1.11.2), `pandas` (1.4.1), `delicatessen` (3.0), `statsmodels` (0.14.0+)

**R: 4.4.1** *(For original `.Rmd` examples)*

* Dependencies: `dplyr` (1.1.4), `geex` (1.1.1), `data.table` (1.15.4)
