import os, json
import numpy as np
from metrics import *
import pandas as pd

def evaluate(data, run_type="single", belief_comb="fair-greedy"):
    # Acceptance Rate, Average Turns, Total proposer payout, Total responder payout, Average of split@1, Average of split@n, Average of split@last, Average downward revs
    acc_rate = acceptance_rate(data, run_type)
    avg_turns = average_turns(data, run_type)
    prop_pay = proposer_payout(data, run_type)
    resp_pay = responder_payout(data, run_type)
    prop_rej_split = rejected_splits(data, run_type, agent=0)
    prop_split_1 = split_amm(data, run_type, turn=0, agent=0)
    # resp_split_1 = split_amm(data, run_type, turn=0, agent=1)
    prop_split_last = split_amm(data, run_type, turn=-1, agent=0)
    # resp_split_last = split_amm(data, run_type, turn=-1, agent=1)
    acc_prop_split_last = acc_split_amm(data, run_type, agent=0)
    acc_resp_split_last = acc_split_amm(data, run_type, agent=1)
    downward_revs = down_revs(data, run_type)
    prop_strats, resp_strats = extract_strats(data, run_type)
    split_c, acc_rates, total_c = split_counts(data)
    first_split_total, first_split_rates = first_split_counts(data)

    prop_ds = deviation_score(data, belief=belief_comb.split('-')[0], agent=0)
    resp_ds = deviation_score(data, belief=belief_comb.split('-')[1], agent=1)
    resp_rej_ds = deviation_score(data, belief=belief_comb.split('-')[1], agent=1, rej=True)
    
    metrics = [acc_rate, avg_turns, (prop_pay, resp_pay), (prop_ds, resp_ds, resp_rej_ds), prop_split_1, prop_split_last, acc_prop_split_last, prop_strats, resp_strats, total_c, acc_rates, first_split_total, first_split_rates]
    return metrics

experiment_map = {
    "belief_private_cot": "CoT",
    "belief_private_tom-zero": "ToM Zero",
    "belief_private_tom-first": "ToM First",
    "belief_private_tom-both": "ToM Both",
    "belief_private_vanilla": "Vanilla"
}

if __name__ == "__main__":
    for model_name in ["gpt-4o"]:
        sim_dir = os.path.join("output", model_name)

        if not os.path.exists(f"output/{model_name}"):
            os.makedirs(f"output/{model_name}")

        for belief_type in os.listdir(sim_dir):
            df_dic = {}
            curr_dir = os.path.join(sim_dir, belief_type)
            for exp_type in os.listdir(curr_dir):
                exp_dir = os.path.join(curr_dir, exp_type)
                data = []
                print(exp_type)
                for runs in os.listdir(exp_dir):
                    curr_run = os.path.join(exp_dir, runs)
                    curr_data = json.load(open(curr_run, "r"))

                    data.append(curr_data)
                print(f"{exp_dir}, {exp_type}")
                metrics = evaluate(data, belief_comb=belief_type)
                df_dic[f"{exp_type}"] = metrics
        
            df = pd.DataFrame.from_dict(df_dic).T
            df.columns = ["Acceptance Rate", "Average Turns", "Total payouts", "Deviation Scores", "Proposer average of split@1", "Proposer average of split@last", "Proposer accepted average of split@last", "Proposer strategies", "Responder strategies", "Total Split Counts", "Split-wise Acc. Rate", "First Split Total", "First Split Rates"]

            df = df.sort_index()

            df.index.name = "Experiment"
            df.reset_index(inplace=True)

            df['Experiment'] = df['Experiment'].map(experiment_map)

            df.to_csv(f"output/{model_name}/{belief_type}.csv")