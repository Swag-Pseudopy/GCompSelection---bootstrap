import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
import warnings

# Import the exact DGM from the repository
from dgm import dgm_example2

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Core Estimators for Case 2
# ---------------------------------------------------------
def fit_glm(formula, data):
    """Helper to fit logistic regression."""
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit(disp=False)

def est_case2(df):
    """Calculates Point Estimates for Case Study 2."""
    res = {}
    df_s1 = df[df['S'] == 1]
    df1, df0 = df.assign(A=1), df.assign(A=0)
    
    # 1. Complete-case analysis (naïve)
    y1_naive = df_s1[df_s1['A'] == 1]['Y'].mean()
    y0_naive = df_s1[df_s1['A'] == 0]['Y'].mean()
    res['Complete-case'] = y1_naive - y0_naive
    
    # 2, 3, 4. Standard G-computation variants
    for name, formula in [('Standard {Z}', "Y ~ A + Z"), 
                          ('Standard {X}', "Y ~ A + X"), 
                          ('Standard {X, Z}', "Y ~ A + X + Z")]:
        m = fit_glm(formula, df_s1)
        res[name] = m.predict(df1).mean() - m.predict(df0).mean()
        
    # 5. Iterated g-computation (Nested)
    m_inner = fit_glm("Y ~ A + X + Z", df_s1)
    df_temp = df.copy()
    df_temp['y1_inner'] = m_inner.predict(df1)
    df_temp['y0_inner'] = m_inner.predict(df0)
    
    m_out1 = fit_glm("y1_inner ~ A + Z", df_temp)
    m_out0 = fit_glm("y0_inner ~ A + Z", df_temp)
    res['Iterated'] = m_out1.predict(df1).mean() - m_out0.predict(df0).mean()
    
    # 6. Inverse probability weighting
    m_s = fit_glm("S ~ X", df)
    m_a = fit_glm("A ~ Z", df)
    
    pi_s = m_s.predict(df)
    pi_a = m_a.predict(df)
    
    ipmw = df['S'] / pi_s
    iptw = df['A'] / pi_a + (1 - df['A']) / (1 - pi_a)
    ipw = ipmw * iptw
    
    wt1, wt0 = df['A'] * ipw, (1 - df['A']) * ipw
    res['IPW'] = (wt1 * df['Y']).sum() / wt1.sum() - (wt0 * df['Y']).sum() / wt0.sum()
    
    return res

# ---------------------------------------------------------
# Bootstrap Variance Estimator (inside Monte Carlo loop)
# ---------------------------------------------------------
def get_bootstrap_ses(df, B=200):
    """Runs a single round of Non-Parametric bootstraps to get SEs."""
    n = len(df)
    boot_dist = {k: [] for k in est_case2(df).keys()}
    
    for _ in range(B):
        df_boot = df.sample(n=n, replace=True)
        for k, v in est_case2(df_boot).items():
            boot_dist[k].append(v)
            
    return {k: np.std(v) for k, v in boot_dist.items()}

# ---------------------------------------------------------
# Main Monte Carlo Simulation Experiment
# ---------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters
    n_runs = 100       # Set to 5000 for the full paper replication
    n_sample = 1000
    n_boot = 200       # Bootstrap resamples per iteration
    
    np.random.seed(7777777)
    
    # Truth calculation based on the Appendix A2.3 of the paper
    # The paper uses a 10M sample to approximate truth, which is -0.205
    truth = -0.205 
    
    # Storage for results
    records = []
    
    print(f"Running Case 2 Monte Carlo Simulation ({n_runs} runs, {n_boot} bootstraps/run)...")
    for i in tqdm(range(n_runs)):
        # 1. Generate new synthetic sample using DGM 2
        d = dgm_example2(n=n_sample, truth=False)
        
        # 2. Calculate Point Estimates
        pt_ests = est_case2(d)
        
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
    
    estimators = ['Complete-case', 'Standard {Z}', 'Standard {X}', 'Standard {X, Z}', 'Iterated', 'IPW']
    
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
    print("\nBootstrap Simulation Results (Case 2):")
    print(df_final.round(3).to_markdown())
