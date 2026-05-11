import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Core Estimators (Mapping estimating equations to sequential GLMs)
# ---------------------------------------------------------
def fit_glm(formula, data):
    """Helper to fit logistic regression."""
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit(disp=False)

def est_case1(df):
    """Table 1: Case 1 Estimators"""
    res = {}
    df_s1 = df[df['S'] == 1]
    
    # 1. Complete-case analysis (naïve)
    y1_naive = df_s1[df_s1['A'] == 1]['Y'].mean()
    y0_naive = df_s1[df_s1['A'] == 0]['Y'].mean()
    res['Complete-case analysis (naïve)'] = y1_naive - y0_naive
    
    # 2 & 3. Standard and Modified g-computation
    m = fit_glm("Y ~ A * W", df_s1)
    df1, df0 = df.assign(A=1), df.assign(A=0)
    
    y1_hat, y0_hat = m.predict(df1), m.predict(df0)
    res['Standard g-computation'] = y1_hat.mean() - y0_hat.mean()
    res['Modified g-computation'] = y1_hat[df['A'] == 1].mean() - y0_hat[df['A'] == 0].mean()
    
    # 4. Inverse probability weighting
    # Selection model based on A and W (treatment induced selection)
    m_s = fit_glm("S ~ A * W", df)
    pi_s = m_s.predict(df)
    ipw = df['S'] / pi_s
    
    # Weighted means
    wt1, wt0 = df['A'] * ipw, (1 - df['A']) * ipw
    res['Inverse probability weighting'] = (wt1 * df['Y']).sum() / wt1.sum() - (wt0 * df['Y']).sum() / wt0.sum()
    
    return res

def est_case2(df):
    """Table 2: Case 2 Estimators"""
    res = {}
    df_s1 = df[df['S'] == 1]
    df1, df0 = df.assign(A=1), df.assign(A=0)
    
    # 1. Complete-case analysis (naïve)
    y1_naive = df_s1[df_s1['A'] == 1]['Y'].mean()
    y0_naive = df_s1[df_s1['A'] == 0]['Y'].mean()
    res['Complete-case analysis (naïve)'] = y1_naive - y0_naive
    
    # 2, 3, 4. Standard G-computation variants
    for name, formula in [('Standard g-computation,{Z}', "Y ~ A + Z"), 
                          ('Standard g-computation,{X}', "Y ~ A + X"), 
                          ('Standard g-computation,{X, Z}', "Y ~ A + X + Z")]:
        m = fit_glm(formula, df_s1)
        res[name] = m.predict(df1).mean() - m.predict(df0).mean()
        
    # 5. Iterated g-computation (Nested)
    m_inner = fit_glm("Y ~ A + X + Z", df_s1)
    df_temp = df.copy()
    df_temp['y1_inner'] = m_inner.predict(df1)
    df_temp['y0_inner'] = m_inner.predict(df0)
    
    m_out1 = fit_glm("y1_inner ~ A + Z", df_temp)
    m_out0 = fit_glm("y0_inner ~ A + Z", df_temp)
    res['Iterated g-computation'] = m_out1.predict(df1).mean() - m_out0.predict(df0).mean()
    
    # 6. Inverse probability weighting
    # Selection depends on X, Treatment depends on Z
    m_s = fit_glm("S ~ X", df)
    m_a = fit_glm("A ~ Z", df)
    
    pi_s = m_s.predict(df)
    pi_a = m_a.predict(df)
    
    # Calculate weights exactly as defined in estfun.py psi_ipw_case2
    ipmw = df['S'] / pi_s
    iptw = df['A'] / pi_a + (1 - df['A']) / (1 - pi_a)
    ipw = ipmw * iptw
    
    wt1, wt0 = df['A'] * ipw, (1 - df['A']) * ipw
    res['Inverse probability weighting'] = (wt1 * df['Y']).sum() / wt1.sum() - (wt0 * df['Y']).sum() / wt0.sum()
    
    return res

# ---------------------------------------------------------
# Bootstrap Wrapper & Table Generation
# ---------------------------------------------------------
def run_comparison_table(df, est_func, table_name, B=500):
    n = len(df)
    m = int(n * 0.9)  # m-out-of-n size
    d = int(n * 0.1)  # delete-d size
    
    pt_ests = est_func(df)
    
    np_boots = {k: [] for k in pt_ests.keys()}
    m_boots = {k: [] for k in pt_ests.keys()}
    jk_boots = {k: [] for k in pt_ests.keys()}
    
    np.random.seed(42)
    print(f"\nRunning Bootstraps for {table_name} (B={B})...")
    for _ in tqdm(range(B)):
        # 1. Non-parametric
        df_np = df.sample(n=n, replace=True)
        for k, v in est_func(df_np).items(): np_boots[k].append(v)
            
        # 2. m-out-of-n
        df_m = df.sample(n=m, replace=True)
        for k, v in est_func(df_m).items(): m_boots[k].append(v)
            
        # 3. Delete-d Jackknife
        df_jk = df.sample(n=n-d, replace=False)
        for k, v in est_func(df_jk).items(): jk_boots[k].append(v)

    # Compile Table Data
    records = []
    for k in pt_ests.keys():
        pt = pt_ests[k]
        np_se = np.std(np_boots[k])
        m_se = np.std(m_boots[k]) * np.sqrt(m/n)
        jk_se = np.std(jk_boots[k]) * np.sqrt((n-d)/d)
        
        records.append({
            'Estimator': k,
            'Point Estimate': pt,
            'NP-Boot SE': np_se,
            'NP 95% CI': f"[{pt - 1.96*np_se:.3f}, {pt + 1.96*np_se:.3f}]",
            'm-Boot SE': m_se,
            'm-Boot CI': f"[{pt - 1.96*m_se:.3f}, {pt + 1.96*m_se:.3f}]",
            'Jackknife SE': jk_se,
            'Jackknife CI': f"[{pt - 1.96*jk_se:.3f}, {pt + 1.96*jk_se:.3f}]"
        })
        
    df_res = pd.DataFrame(records).set_index('Estimator')
    print(f"\n=== {table_name} ===")
    print(df_res.round(3).to_markdown())

# Execution
if __name__ == "__main__":
    d1 = pd.read_csv("data/example1.csv")
    d2 = pd.read_csv("data/example2.csv")
    
    run_comparison_table(d1, est_case1, "Table 1: Case 1 Results", B=500)
    run_comparison_table(d2, est_case2, "Table 2: Case 2 Results", B=500)
