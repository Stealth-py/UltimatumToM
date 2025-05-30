import numpy as np
import re

MAX_TURNS = 5

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
    if ":" not in reasoning:
        return reasoning.strip("Strategy ").strip('[').strip('(')[0]
    else:
        return reasoning.strip("Strategy: ").strip('[').strip('(')[0]
    # .split(')')[0].split('\n')[0].split()[0].strip('.').strip('[').strip(']')

def deviation_score(data, belief, agent=0, rej=False):
    expected_share = []
    if agent == 0:
        if belief == "fair":
            expected_share = [5]
        elif belief == "greedy":
            expected_share = [7, 8, 9, 10]
        elif belief == "selfless":
            expected_share = [0, 1, 2, 3]
    else:
        if belief == "fair":
            expected_share = [5]
        elif belief == "greedy":
            expected_share = [6, 7, 8, 9, 10]
        elif belief == "selfless":
            expected_share = [0, 1, 2, 3, 4]

    deviation = []
    for run in data:
        devs = []
        for turn in run:
            if agent == 0:
                split = extract_split(turn["proposer"])
                share = split[0]
                if share in expected_share:
                    deviation.append(0)
                else:
                    deviation.append(min(abs(share - min(expected_share)), abs(share - max(expected_share))))
                break
            else:
                if not rej:
                    if "accept" in turn["responder"].lower():
                        split = extract_split(turn["proposer"])
                        share = split[1]
                        if share in expected_share:
                            deviation.append(0)
                        else:
                            deviation.append(min(abs(share - min(expected_share)), abs(share - max(expected_share))))
                        break
                else:
                    if "accept" not in turn["responder"].lower():
                        split = extract_split(turn["proposer"])
                        share = split[1]
                        if share in expected_share:
                            devs.append(0)
                        else:
                            devs.append(min(abs(share - min(expected_share)), abs(share - max(expected_share))))
        # print(deviation, belief)
        
        if rej:
            if devs != []:
                deviation.append(np.mean(devs))
            else:
                deviation.append(-1)

    deviation = [i for i in deviation if i != -1]
    
    if deviation == []:
        return -1
    return round(np.mean(deviation), 2)

def extract_strats(data, run_type):
    prop_strats = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    }

    resp_strats = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
    }

    if run_type == "single":
        for run in data:
            for turn in run:
                # print(turn["proposer_strat"], turn["responder_strat"], turn["turn"], )
                prop_strategy = int(strategy_split(turn["proposer_strat"].replace('*', '')))
                resp_strategy = int(strategy_split(turn["responder_strat"].replace('*', '')))

                prop_strats[prop_strategy] += 1
                resp_strats[resp_strategy] += 1
    else:
        for run in data:
            for turn in run:
                prop_strategy = int(strategy_split(turn["proposer_strat"].replace('*', '')))
                resp_strategy = int(strategy_split(turn["responder_strat"].replace('*', '')))

                prop_strats[prop_strategy] += 1
                resp_strats[resp_strategy] += 1

    return prop_strats, resp_strats

def acceptance_rate(data, run_type):
    if run_type == "single":
        c = 0
        for run in data:
            if "accept" in run[-1]["responder"].lower():
                c+=1
        
        return round(100 * c / len(data), 2)
    else:
        counts = []
        for run in data:
            c = 0
            for turn in run:
                if "accept" in turn["responder"].lower():
                    c+=1
            counts.append(100 * c/len(run))
        return round(np.mean(counts), 2)

def average_turns(data, run_type):
    if run_type == "single":
        c = 0
        for run in data:
            c+= len(run)
        return round(c / len(data), 2)
    else:
        counts = []
        for run in data:
            c = 0
            for turn in run:
                c+=1
                if "accept" in turn["responder"].lower():
                    counts.append(c)
                    c = 0
            if c != 0:
                counts.append(c)
        return round(np.mean(counts), 2)

def proposer_payout(data, run_type):
    if run_type == "single":
        payout = 0
        for run in data:
            if "accept" in run[-1]["responder"].lower():
                payout += extract_split(run[-1]["proposer"])[0]
        return payout
    else:
        payouts = 0
        max_pay = 0
        turns = []
        for run in data:
            payout = 0
            c = 0
            for turn in run:
                c += 1
                if "accept" in turn["responder"].lower():
                    payout += extract_split(turn["proposer"])[0]
                    max_pay += 10
            payouts+=payout
            turns.append(c)
        
        if turns == 0:
            return 0
        return payouts

def responder_payout(data, run_type):
    if run_type == "single":
        payout = 0
        for run in data:
            if "accept" in run[-1]["responder"].lower():
                payout += extract_split(run[-1]["proposer"])[1]
        return payout
    else:
        payouts = 0
        max_pay = 0
        turns = []
        for run in data:
            payout = 0
            c = 0
            for turn in run:
                c += 1
                if "accept" in turn["responder"].lower():
                    payout += extract_split(turn["proposer"])[1]
                    max_pay += 10
            payouts+=payout
            turns.append(c)
        
        if turns == 0:
            return 0
        return payouts 

def split_amm(data, run_type, turn=0, agent=0):
    # amm -> avg, min, max
    # agent 0 -> proposer; agent 1 -> responder
    if run_type == "single" or "mturn" not in run_type:
        splits = []
        for run in data:
            splits.append(extract_split(run[turn]["proposer"])[agent])
        splits = np.array(splits)
        return "{} ({}, {})".format(round(np.mean(splits), 2), np.min(splits), np.max(splits))
    else:
        splits = []
        for run in data:
            splits.append(extract_split(run[0]["proposer"])[agent])
            for tr in range(1, len(run)):
                # if tr==1:
                #     splits.append(extract_split(run[tr-1]["proposer"])[agent])
                #     continue
                if "accept" in run[tr-1]["responder"].lower():
                    if turn == -1:
                        splits.append(extract_split(run[tr-1]["proposer"])[agent])
                    else:
                        splits.append(extract_split(run[tr]["proposer"])[agent])
                    continue
            if turn == -1:
                splits.append(extract_split(run[-1]["proposer"])[agent])
        splits = np.array(splits)
        return "{} ({}, {})".format(round(np.mean(splits), 2), np.min(splits), np.max(splits))

def rejected_splits(data, run_type, agent=0):
    splits = []
    for run in data:
        for turn in range(len(run)-1):
            if "accept" not in run[turn]["responder"].lower():
                splits.append(extract_split(run[turn]["proposer"])[agent])
    if splits == []:
        return (0, 0, 0)
    splits = np.array(splits)
    return "{} ({}, {})".format(round(np.mean(splits), 2), np.min(splits), np.max(splits))

def split_counts(data):
    acc_counts = {}
    total_counts = {}

    for run in data:
        for turn in range(len(run)):
            split = extract_split(run[turn]["proposer"])

            if "accept" in run[turn]["responder"].lower():
                if (split[0], split[1]) not in acc_counts:
                    acc_counts[(split[0], split[1])] = 1
                else:
                    acc_counts[(split[0], split[1])] += 1
            
            if (split[0], split[1]) not in total_counts:
                total_counts[(split[0], split[1])] = 1
            else:
                total_counts[(split[0], split[1])] += 1
    
    split_acc_rates = {}
    for key in acc_counts:
        split_acc_rates[key] = round(acc_counts[key] / total_counts[key], 2)

    return acc_counts, split_acc_rates, total_counts

def first_split_counts(data):
    acc_counts = {}
    total_counts = {}

    for run in data:
        split = extract_split(run[0]["proposer"])
        
        if (split[0], split[1]) not in total_counts:
            total_counts[(split[0], split[1])] = 1
        else:
            total_counts[(split[0], split[1])] += 1
        
        if "accept" in run[0]["responder"].lower():
            if (split[0], split[1]) not in acc_counts:
                acc_counts[(split[0], split[1])] = 1
            else:
                acc_counts[(split[0], split[1])] += 1
        
        for tr in range(1, len(run)):
            if "accept" in run[tr-1]["responder"].lower():
                # split_last = extract_split(run[tr]["proposer"])
                split = extract_split(run[tr]["proposer"])
            
                if (split[0], split[1]) not in total_counts:
                    total_counts[(split[0], split[1])] = 1
                else:
                    total_counts[(split[0], split[1])] += 1
                
                if "accept" in run[tr]["responder"].lower():
                    if (split[0], split[1]) not in acc_counts:
                        acc_counts[(split[0], split[1])] = 1
                    else:
                        acc_counts[(split[0], split[1])] += 1

    for key in acc_counts:
        acc_counts[key] = round(acc_counts[key] / total_counts[key], 2)

    return total_counts, acc_counts


def acc_split_amm(data, run_type, agent=0):
    # amm -> avg, min, max
    # agent 0 -> proposer; agent 1 -> responder
    if run_type == "single":
        splits = []
        for run in data:
            if "accept" in run[-1]["responder"].lower():
                splits.append(extract_split(run[-1]["proposer"])[agent])
        if splits == []:
            return (0, 0, 0)
        splits = np.array(splits)
        return "{} ({}, {})".format(round(np.mean(splits), 2), np.min(splits), np.max(splits))
    else:
        splits = []
        for run in data:
            for turn in run:
                if "accept" in turn["responder"].lower():
                    splits.append(extract_split(turn["proposer"])[agent])
        if splits == []:
            return (0, 0, 0)
        splits = np.array(splits)
        return "{} ({}, {})".format(round(np.mean(splits), 2), np.min(splits), np.max(splits))

def down_revs(data, run_type):
    if run_type == "single":
        revs = 0
        for run in data:
            for turn in range(len(run)-1):
                curr_prop = extract_split(run[turn]["proposer"])[-1]
                next_prop = extract_split(run[turn+1]["proposer"])[-1]

                if next_prop > curr_prop:
                    revs+=1
        
        return round(revs / len(data), 2)
    else:
        revs = 0
        c = 0
        for run in data:
            for turn in range(len(run)-1):
                if "accept" not in run[turn]["responder"].lower():
                    c+=1
                    curr_prop = extract_split(run[turn]["proposer"])[-1]
                    next_prop = extract_split(run[turn+1]["proposer"])[-1]

                    if next_prop > curr_prop:
                        revs+=1

        if c==0:
            return 0
        return round(revs / c, 2)