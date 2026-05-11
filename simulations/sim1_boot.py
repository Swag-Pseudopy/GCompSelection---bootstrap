import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
import warnings

# Import the exact DGM from the repository
from dgm import dgm_example1 

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Core Estimators
# ---------------------------------------------------------
def fit_glm(formula, data):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit(disp=False)

def est_case1(df):
    """Calculates Point Estimates for Case 1."""
    res = {}
    df_s1 = df[df['S'] == 1]
    
    # 1. Complete-case
    res['Complete-case'] = df_s1[df_s1['A'] == 1]['Y'].mean() - df_s1[df_s1['A'] == 0]['Y'].mean()
    
    # 2 & 3. Standard & Proposed (Modified)
    m = fit_glm("Y ~ A * W", df_s1)
    df1, df0 = df.assign(A=1), df.assign(A=0)
    y1_hat, y0_hat = m.predict(df1), m.predict(df0)
    
    res['Standard'] = y1_hat.mean() - y0_hat.mean()
    res['Proposed'] = y1_hat[df['A'] == 1].mean() - y0_hat[df['A'] == 0].mean()
    
    # 4. IPW
    m_s = fit_glm("S ~ A * W", df)
    ipw = df['S'] / m_s.predict(df)
    wt1, wt0 = df['A'] * ipw, (1 - df['A']) * ipw
    res['IPW'] = (wt1 * df['Y']).sum() / wt1.sum() - (wt0 * df['Y']).sum() / wt0.sum()
    
    return res

# ---------------------------------------------------------
# Bootstrap Variance Estimator (inside Monte Carlo loop)
# ---------------------------------------------------------
def get_bootstrap_ses(df, B=200):
    """Runs a single round of Non-Parametric bootstraps to get SEs."""
    n = len(df)
    boot_dist = {k: [] for k in est_case1(df).keys()}
    
    for _ in range(B):
        df_boot = df.sample(n=n, replace=True)
        for k, v in est_case1(df_boot).items():
            boot_dist[k].append(v)
            
    return {k: np.std(v) for k, v in boot_dist.items()}

# ---------------------------------------------------------
# Main Monte Carlo Simulation Experiment
# ---------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters
    n_runs = 100       # Monte Carlo iterations (Paper uses 5000)
    n_sample = 1000    # Sample size per iteration
    n_boot = 200       # Bootstrap resamples per iteration
    
    np.random.seed(7777777)
    truth = -0.218749  # True value established in sim_case-study1.py
    
    # Storage for results
    records = []
    
    print(f"Running Monte Carlo Simulation ({n_runs} runs, {n_boot} bootstraps/run)...")
    for i in tqdm(range(n_runs)):
        # 1. Generate new synthetic sample
        d = dgm_example1(n=n_sample, truth=False)
        d['AW'] = d['A'] * d['W']
        
        # 2. Calculate Point Estimates
        pt_ests = est_case1(d)
        
        # 3. Calculate Standard Errors via Bootstrap
        boot_ses = get_bootstrap_ses(d, B=n_boot)
        
        # 4. Record iteration metrics
        row = {'iteration': i}
        for est_name in pt_ests.keys():
            est = pt_ests[est_name]
            se = boot_ses[est_name]
            
            row[f'{est_name}_est'] = est
            row[f'{est_name}_se'] = se
            
            # Check Coverage: Is truth inside 95% CI?
            lower, upper = est - 1.96*se, est + 1.96*se
            row[f'{est_name}_cover'] = 1 if (lower < truth < upper) else 0
            
        records.append(row)

    # ---------------------------------------------------------
    # Post-Processing: Calculate Frequentist Properties
    # ---------------------------------------------------------
    df_res = pd.DataFrame(records)
    final_table = []
    
    estimators = ['Complete-case', 'Standard', 'Proposed', 'IPW']
    
    for est in estimators:
        bias = (df_res[f'{est}_est'] - truth).mean()
        ese = df_res[f'{est}_est'].std()
        rmse = np.sqrt(((df_res[f'{est}_est'] - truth)**2).mean())
        mean_se = df_res[f'{est}_se'].mean()
        ser = mean_se / ese
        coverage = df_res[f'{est}_cover'].mean()
        
        final_table.append({
            'estimator': est,
            'bias': bias,
            'ese': ese,
            'rmse': rmse,
            'ser': ser,
            'coverage': coverage
        })
        
    df_final = pd.DataFrame(final_table).set_index('estimator')
    
    print(f"\nTruth: {truth}")
    print("\nBootstrap Simulation Results:")
    print(df_final.round(3))
