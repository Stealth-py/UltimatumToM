import pandas as pd
import os, sys, re, json
from ast import literal_eval
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

def extract_split(proposal):
    # print(proposal.split('\n'))
    proposal = proposal.replace('*', '')
    # print(proposal)
    if '</think>' in proposal:
        proposal = proposal.split('</think>')[-1].split('\n')[-1]
    splits = re.findall(r'\d+\.\d+|\d+', proposal)
    splits = [float(i) for i in splits]
    # print(splits)
    return splits

def strategy_split(reasoning):
    reasoning = reasoning.replace('*', '').replace('[', '').replace('(', '')
    try:
        return re.findall(r'Strategy: \d', reasoning)[0].replace('Strategy: ', '')
    except:
        return re.findall(r'Strategy \d', reasoning)[0].replace('Strategy ', '')
    
def get_strategy_data():
    prop_df = pd.read_csv('outs/combined_proposer_strategies.csv')
    resp_df = pd.read_csv('outs/combined_responder_strategies.csv')

    ds_p = []
    ds_a_r = []
    ds_r_r = []

    for model in os.listdir('outs'):
        if not os.path.isdir(f'outs/{model}'):
            continue
        if '.csv' in model:
            continue
        for prop_belief, resp_belief in [('greedy', 'fair'), ('fair', 'greedy'), ('greedy', 'greedy'), ('selfless', 'greedy'), ('selfless', 'fair'), ('greedy', 'selfless'), ('fair', 'fair'), ('fair', 'selfless'), ('selfless', 'selfless')]:
            if prop_belief == "fair":
                prop_exp = [5]
            elif prop_belief == "greedy":
                prop_exp = [7, 8, 9, 10]
            elif prop_belief == "selfless":
                prop_exp = [0, 1, 2, 3]
            
            if resp_belief == "fair":
                resp_exp = [5]
            elif resp_belief == "greedy":
                resp_exp = [6, 7, 8, 9, 10]
            elif resp_belief == "selfless":
                resp_exp = [0, 1, 2, 3, 4]

            dir_ = f'simulations/{model}/{prop_belief}-{resp_belief}'
            for exp_type in os.listdir(dir_):
                curr_dir = os.path.join(dir_, exp_type)
                # print(curr_dir)
                if not os.path.isdir(curr_dir):
                    continue
                for run in os.listdir(curr_dir):
                    if '.json' not in run:
                        continue
                    curr_d = json.load(open(os.path.join(curr_dir, run), 'r'))

                    first_split = extract_split(curr_d[0]["proposer"])[0]
                    dev_prop = min(abs(first_split - min(prop_exp)), abs(first_split - max(prop_exp)))
                    
                    prop_strat = strategy_split(curr_d[0]['proposer_strat'])

                    ds_p.append([dev_prop, model, prop_belief, resp_belief, 'Strategy ' + prop_strat, exp_type.replace('belief_private_', '')])

                    for turn in curr_d:
                        resp_strat = strategy_split(turn['responder_strat'])

                        if "accept" in turn["responder"].lower():
                            acc_split = extract_split(turn['proposer'])[-1]
                            if prop_belief == "fair":
                                if max(resp_exp) >= acc_split >= min(resp_exp):
                                    dev_resp = 0
                                else:
                                    dev_resp = min(abs(acc_split - min(resp_exp)), abs(acc_split - max(resp_exp)))
                                ds_a_r.append([dev_resp, model, prop_belief, resp_belief, 'Strategy ' + resp_strat, exp_type.replace('belief_private_', '')])
                            else:
                                if max(resp_exp) >= acc_split >= min(resp_exp):
                                    dev_resp = 0
                                else:
                                    dev_resp = min(abs(acc_split - min(resp_exp)), abs(acc_split - max(resp_exp)))
                                ds_a_r.append([dev_resp, model, prop_belief, resp_belief, 'Strategy ' + resp_strat, exp_type.replace('belief_private_', '')])
                            break
                        else:
                            rej_split = extract_split(turn['proposer'])[-1]
                            if resp_belief == "fair":
                                if max(resp_exp) >= rej_split >= min(resp_exp):
                                    dev_resp = 0
                                else:
                                    dev_resp = min(abs(rej_split - min(resp_exp)), abs(rej_split - max(resp_exp)))
                            else:    
                                if max(resp_exp) >= rej_split >= min(resp_exp):
                                    dev_resp = 0
                                else:
                                    dev_resp = min(abs(rej_split - min(resp_exp)), abs(rej_split - max(resp_exp)))
                            ds_r_r.append([dev_resp, model, prop_belief, resp_belief, 'Strategy ' + resp_strat, exp_type.replace('belief_private_', '')])
    
    df_p = pd.DataFrame(ds_p, columns=['DS', 'Model', 'Prop_Belief', 'Resp_Belief', 'Strategy', 'Reasoning'])
    df_a_r = pd.DataFrame(ds_a_r, columns=['DS', 'Model', 'Prop_Belief', 'Resp_Belief', 'Strategy', 'Reasoning'])
    df_r_r = pd.DataFrame(ds_r_r, columns=['DS', 'Model', 'Prop_Belief', 'Resp_Belief', 'Strategy', 'Reasoning'])

    return df_p, df_a_r, df_r_r

# -------------------------------------------------------------------------
# 1. Define the "Theoretical Consistency" Baselines
# -------------------------------------------------------------------------
# This dictionary maps the (Role, Belief) to the Strategy that serves as the
# reference baseline (Equation: beta_target - beta_baseline = 0)
# Note: These should match the "Expected Behavior" defined in the methodology.

baselines = {
    # --- Proposer Strategies ---
    "Proposer": {
        "fair": ["Strategy 3"],
        "greedy": ["Strategy 1", "Strategy 2"],
        "selfless": ["Strategy 4", "Strategy 5"] # or "very generously"
    },

    # --- Responder Acceptance Strategies ---
    "Responder_Accept": {
        "fair": ["Strategy 2"],
        "greedy": ["Strategy 1"],
        "selfless": ["Strategy 3"]
    },

    # --- Responder Rejection Strategies ---
    "Responder_Reject": {
        "fair": ["Strategy 4", "Strategy 6"], # Fair agents reject unfairness
        "greedy": ["Strategy 5", "Strategy 6"], 
        "selfless": ["Strategy 4", "Strategy 5"] # Selfless agents might reject to give more?
    }
}

# -------------------------------------------------------------------------
# 2. Automated Contrast Generator Function
# -------------------------------------------------------------------------
def run_contrasts(df, role_type, belief_col, strategy_col, metric_col):
    """
    Fits the Cell Means model and runs contrasts against the baselines.

    df: DataFrame containing the data
    role_type: Key for the baselines dict (e.g., 'Proposer')
    belief_col: Name of column with Beliefs (e.g., 'Proposer_Belief')
    strategy_col: Name of column with Strategies (e.g., 'Selected_Strategy')
    metric_col: Name of column with Deviation Scores (e.g., 'DS_Proposer')
    """

    print(f"--- Analyzing {role_type} ---")

    # A. Fit Cell Means Model (No Intercept)
    formula = f"{metric_col} ~ 0 + {belief_col}:{strategy_col}"
    model = smf.ols(formula, data=df).fit()

    # B. Generate Hypotheses
    hypotheses = []
    labels = []

    # Get all strategies present in this specific dataset subset
    present_strategies = df[strategy_col].unique()

    for belief, baseline_strat in baselines[role_type].items():
        for strat in present_strategies:
            # Skip baseline (Contrast is 0)
            for bs in baseline_strat:
                if strat == bs:
                    continue

                # Skip if this specific combination doesn't exist in data
                # (e.g. No Greedy agents chose "Selfless" strategy)
                # We check if the coefficient exists in the model params
                param_name = f"{belief_col}[{belief}]:{strategy_col}[{strat}]"
                base_name = f"{belief_col}[{belief}]:{strategy_col}[{bs}]"

                if param_name in model.params and base_name in model.params:
                    # Construct Test: Current - Baseline = 0
                    test_str = f"{param_name} = {base_name}"
                    hypotheses.append(test_str)
                    labels.append(f"[{belief}] {strat} vs {bs}")

    # C. Run t-tests
    if hypotheses:
        results = model.t_test(hypotheses)
        print(results.summary(xname=labels))
    else:
        print("No valid contrasts found (check data/strategy names).")
    print("\n")

# -------------------------------------------------------------------------
# 3. Execution Example (Pseudo-code)
# -------------------------------------------------------------------------
# Assuming dataset is split or has columns for different DS types

if __name__ == "__main__":
    df_p, df_a_r, df_r_r = get_strategy_data()

    # 1. Proposer Analysis
    run_contrasts(df_p, "Proposer", "Prop_Belief", "Strategy", "DS")

    run_contrasts(df_a_r, "Responder_Accept", "Resp_Belief", "Strategy", "DS")

    run_contrasts(df_r_r, "Responder_Reject", "Resp_Belief", "Strategy", "DS")