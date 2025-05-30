import argparse
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
import json
import time
import os

from dotenv import load_dotenv
load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

"""
Multi-turn: true or false (single-turn games or multi-turn games)
Stake type: single or cumulative (single-stake games or multi-stake games)
Belief: greedy, fair, selfless (will always be true)
Inference visibility: public or private (public or private beliefs)
(ToM) Reasoning can be: using CoT (not ToM; default case), zero (only self), first (other), both (both self and other)
Strategic labeling: will always be true (it shall be done with the model decisions: accept-reject or proposals)
Post-hoc explanation: it shall be done after the end of 1 run

Current pipeline of prompts:
(1) proposer/responder instruction prompt (with belief prompt) -> (2) CoT/tom reasoning prompt (zero-order/1st-order/both) -> LLM -> (3) generated split/accept-reject + strategic labeling prompt -> LLM -> end or goto (2)

All simulations are multi-turn single-stake games with belief.
"""

def simulations():
    cfg = json.load(open("conf/cfg.json"))
    for prop_belief, resp_belief in [('greedy', 'fair'), ('fair', 'greedy'), ('greedy', 'greedy'), ('selfless', 'greedy'), ('selfless', 'fair'), ('greedy', 'selfless'), ('fair', 'fair'), ('fair', 'selfless'), ('selfless', 'selfless')]:
        cfg["prop_belief"] = prop_belief
        cfg["resp_belief"] = resp_belief

        for inf_visibility in ['private']:
            cfg["inf_visibility"] = inf_visibility

            for tom in ["cot", "zero", "first", "both", "vanilla"]:
                cfg["tom_type"] = tom
                simulation(cfg)


"""
The config might be a bit weird to make sense of at first, but here are the general definitions of each variable in the config.

Environment-specific:
-> `model_name`: the model you want to perform the simulation for. there is no enum for it, specifically, but if you want to be consistent with our setup then you can choose from one of the following models: claude-3-5-haiku-20241022, deepseek-r1-distill-qwen-32b, gpt-4o, gpt-4o-mini, llama-3.1-8b-instant, llama-3.3-70b-versatile, o3-mini
-> `temperature`: the temperature you want to use for inference.
-> `output_directory`: directory where you want to output the logs and results of the simulation

Game-specific:
-> `max_turns`: maximum number of turns for which one run should run for.
-> `num_sims`: number of simulations we want to perform for each experiment.
-> `stake`: we can vary the amount of money, that is stake, which will be divided by the two agents. we wanted to include this analysis as well but it was already a bit too compressed.
-> `multi_turn`: this will always be true, and is not used in this code. earlier we wanted to compare with single-turn experiments.
-> `stake_type`: this will always be single, and is not used in this code. earlier we wanted to compare with experiments where each simulation can have multiple sub-simulation.
-> `do_belief`: should always be true, unless you are doing an experiment for no beliefs in agents. this would override the `prop_belief`, and `resp_belief` arguments.
-> `tom_type`: this will either be cot, zero, first, both, or vanilla. the naming is weird for the variables but this is because i did not want to separate them for the sake of simplicity :)
-> `prop_belief`: this will either be selfless, fair, or greedy.
-> `resp_belief`: this will either be selfless, fair, or greedy.
-> `inf_visibility`: this should always be private. earlier we wanted to compare with simulations where each agent is able to see some private information of the other agent.

"""


def simulation(cfg):
    num_sims = cfg["num_sims"]
    dirname = cfg["output_directory"]
    filename = ""

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists(os.path.join(dirname, cfg["model_name"])):
        os.makedirs(os.path.join(dirname, cfg["model_name"]))

    dirname = os.path.join(dirname, cfg["model_name"])

    exp_type = f"{cfg['prop_belief']}-{cfg['resp_belief']}"

    tail_filename = ''

    if not cfg["do_belief"]:
        exp_type = "vanilla"
        filename = "vanilla"
    else:
        filename = "belief"
        filename += f"_{cfg['inf_visibility']}"
    
    if cfg["tom_type"] in ["zero", "first", "both"]:
        if not filename == "":
            filename += "_tom-" + cfg['tom_type']
        else:
            filename = "tom-" + cfg['tom_type']
    else:
        filename += "_" + cfg['tom_type']
    

    out_dir = os.path.join(dirname, exp_type)
    out_dir = os.path.join(out_dir, filename+tail_filename)

    print(out_dir, filename, tail_filename)
    
    for i in range(num_sims):
        # Output the dialogue history to a text file
        dialogue_hist, chat_logs = experiment(cfg)
        
        out_file = os.path.join(out_dir, f"{i+1}.json")

        log_dir = os.path.join("logs", exp_type)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = os.path.join(log_dir, f"{filename+tail_filename}.log")
        
        print(f"\nDialogue history saved to {out_file}")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(out_file, "w") as f:
            json.dump(dialogue_hist, f)
        
        with open(log_dir, "a") as f:
            f.write("Simulation " + str(i+1) + "\n")
            for log in chat_logs:
                f.write(log + "\n")
            f.write("=======================================================\n")


def experiment(cfg):
    stake = cfg['stake']

    proposer_name = "A"
    responder_name = "B"

    max_turns = cfg["max_turns"]

    api = "closed"

    if "gpt" in cfg["model_name"] or "o3-mini" in cfg["model_name"]:
        if "o3-mini" in cfg["model_name"]:
            cfg["temperature"] = 1.0
        llm1 = ChatOpenAI(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("OPENAI"))
        llm2 = ChatOpenAI(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("OPENAI"))
    elif "gemini" in cfg["model_name"]:
        llm1 = ChatGoogleGenerativeAI(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("GOOGLE"))
        llm2 = ChatGoogleGenerativeAI(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("GOOGLE"))
    elif "claude" in cfg["model_name"]:
        llm1 = ChatAnthropic(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("ANTHROPIC"), rate_limiter=rate_limiter)
        llm2 = ChatAnthropic(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("ANTHROPIC"), rate_limiter=rate_limiter)
    elif "llama" in cfg["model_name"] or "deepseek" in cfg["model_name"]:
        api="open"
    
    if api == "open":
        llm1 = ChatGroq(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("GROQ"))
        llm2 = ChatGroq(model=cfg["model_name"], temperature=cfg["temperature"], api_key=os.getenv("GROQ"))

    # Base rationale for each agent similar to that given to human subjects
    base_rationale = """
    Your goal is to maximize your share of the money. Prioritize being very consistent with your beliefs while making a decision.
    """

    ''' Personal belief
    ---
    To-do: Ideas for exploration. Need to survey literature.:
    1. Agent's belief system (part of persona-based role playing):
    fairness vs greediness
    altruistic vs. selfish
    risk averse vs risk taking
    individualism vs collectivism
    pragmistic vs. idealistic
    optimistic vs. pessimistic
    logical vs. emotional
    2. Belief incongruency
    3. Personal traits (OCEAN model). Already done in Noh and Change (2024). Could be incorporated with ToM.
    '''

    if not cfg["do_belief"]:
        proposer_belief = ""
        responder_belief = ""
    else:

        common_belief = "A greedy person is someone who is selfish and only cares about themselves. A fair person is someone who is willing to share equally. A selfless person is someone who is willing to give everything to others. "

        proposer_belief = common_belief
        responder_belief = common_belief

        if cfg['inf_visibility'] == "public":
            proposer_belief += f"Player {responder_name} knows that you are {cfg['prop_belief']} and you know that Player {responder_name} is {cfg['resp_belief']}."
            responder_belief += f"Player {proposer_name} knows that you are {cfg['resp_belief']} and you know that Player {proposer_name} is {cfg['prop_belief']}."
        else:
            proposer_belief += f"You are {cfg['prop_belief']}."
            responder_belief += f"You are {cfg['resp_belief']}."
    
    proposer_strat = f"""Given the context of the game, choose one of the following strategies for your proposal that best aligns with your perspective as a proposer.
    1) Propose very greedily
    2) Propose greedily
    3) Propose fairly
    4) Propose generously
    5) Propose very generously
    6) Other
    """
    
    responder_strat = """Given the context of the game, choose one of the following strategies for your decision that best aligns with your perspective as a responder.
    1) Accept a favourable offer;
    2) Accept a fair offer;
    3) Accept an unfavourable offer;
    4) Reject a favourable offer;
    5) Reject a fair offer;
    6) Reject an unfavourable offer;
    7) Other
    """

    if not cfg["post-response_reflection"]:
        proposer_expl = ""
        responder_expl = ""
    else:
        proposer_expl = """
        Reflect on your decision in the game. Explain your thought process and strategy step by step.
        """
        responder_expl = """
        Reflect on your decision in the game. Explain your thought process and strategy step by step.
        """

    # Guided ToM prompting

    if cfg["tom_type"] == "cot":
        proposer_tom = """To achieve your goal, let's think step-by-step. Output your chain-of-thought."""
        responder_tom = """To achieve your goal, let's think step-by-step. Output your reasoning."""

    elif cfg['tom_type'] == "zero":
        proposer_tom = f"""To achieve your goal, answer questions about your own state of mind, given the current conversation:
        1. What are your beliefs?
        2. What are you desires in this situation?
        3. What are your intentions for this situation?"""
        responder_tom = f"""To achieve your goal, answer questions about your own state of mind, given the current conversation:
        1. What are your beliefs?
        2. What are you desires in this situation?
        3. What are your intentions for this situation?"""

    elif cfg['tom_type'] == "first":
        proposer_tom = f"""To achieve your goal, talk about the state of mind of Player {responder_name}, given the current conversation:
        1. What do you think Player {responder_name}'s beliefs are?
        2. WHat do you think Player {responder_name}'s desires are?
        3. What do you think Player {responder_name}'s intentions are for this situation?"""
        responder_tom = f"""To achieve your goal, talk about the state of mind of Player {proposer_name}, given the current conversation:
        1. What do you think Player {proposer_name}'s beliefs are?
        2. WHat do you think Player {proposer_name}'s desires are?
        3. What do you think Player {proposer_name}'s intentions are for this situation?"""

    elif cfg['tom_type'] == "both":
        proposer_tom = f"""To achieve your goal, talk about both your and Player {responder_name}'s state of mind, given the current conversation:
        1. What are your beliefs?
        2. What do you think Player {responder_name}'s beliefs are?
        3. What are you desires in this situation?
        4. What do you think Player {responder_name}'s desires are?
        5. What are your intentions for this situation?
        6. What do you think Player {responder_name}'s intentions are for this situation?"""
        responder_tom = f"""To achieve your goal, talk about both your and Player {proposer_name}'s state of mind, given the current conversation:
        1. What are your beliefs?
        2. What do you think Player {proposer_name}'s beliefs are?
        3. What are you desires in this situation?
        4. What do you think Player {proposer_name}'s desires are?
        5. What are your intentions for this situation?
        6. What do you think Player {proposer_name}'s intentions are for this situation?"""
    else:
        proposer_tom = ""
        responder_tom = ""

    # Putting things together in final instructions
    instr_llm1 = f"""
    You are Player {proposer_name} in a negotiation game. There is ${stake} to split between you and Player {responder_name}.
    {proposer_belief}
    {base_rationale}
    Propose an offer to Player {responder_name} (Simply say: I get $X and you get $Y.).
    The game only ends when the responder says `Accept`. If the responder says `Reject`, you will need to make a new proposal. Payoffs are calculated at the end of the game. You need to choose a strategy that best fits the situation. Only answer what is asked of you at each point.
    IMPORTANT: The game will only end when Player {responder_name} accepts your offer or the maximum number of rounds is reached. Think about your payoff if your offer is rejected. Both players get 0 payoff if the game ends without an accepted offer.
    Your beliefs are yours alone, and you are required to keep them private.
    """

    reason_prmp = ""

    if cfg["tom_type"] in ["cot", "zero", "first", "both"]:
        instr_llm1 += " Only talk about your beliefs and ToM with the mediator when asked to. That conversation will be private. At the start of every round you need to perform reasoning first based on the following instructions. Unless explicitly asked to make a proposal, you should not make a proposal. Only answer what is asked of you at each point."

        reason_prmp += "Make your decision based on the reasoning you provided earlier."

    instr_llm2 = f"""
    You are Player {responder_name} in a negotiation game. Player {proposer_name} will propose how to split ${stake} between you.
    {responder_belief}
    {base_rationale}
    You can either accept or reject their offer. Rejecting the offer does not end the game.
    Always write whether you `Accept` or `Reject` the offer. Accepting the offer will end the game and your payoff will be calculated. You need to choose a strategy that best fits the situation. Only answer what is asked of you at each point.
    IMPORTANT: The game will only end when you accept Player {proposer_name}'s offer or the maximum number of rounds is reached. Think about your payoff if your offer is rejected. Both players get 0 payoff if the game ends without an accepted offer.
    Your beliefs are yours alone, and you are required to keep them private.
    """

    if cfg["tom_type"] in ["cot", "zero", "first", "both"]:
        instr_llm2 += " Only talk about your beliefs and ToM with the mediator when asked to. That conversation will be private. At the start of every round you need to perform reasoning first based on the following instructions. Unless explicitly asked to respond to the proposal, you should not make a decision. Only answer what is asked of you at each point."

    prop_decision_prompt = f"""Propose an offer to Player {responder_name}. Simply say 'I get $X and you get $Y.' {reason_prmp}
    {proposer_strat}

    Strictly follow the format- Proposal: [proposal] | Strategy [number]. Make your proposal here. Do not write anything other than the format.
    """

    resp_decision_prompt = f"""Respond to the proposal made by Player {proposer_name}. {reason_prmp}
    {responder_strat}

    Strictly follow the format- Decision: [decision] | Strategy [number]. Make your decision here. Do not write anything other than the format.
    """

    print(f"Running a negotiation game with the following params: {cfg}")

    # Initial negotiation states
    llm1_messages = [HumanMessage(content=instr_llm1)]
    llm2_messages = [HumanMessage(content=instr_llm2)]

    # Store dialogue history here
    dialogue_hist = []
    chat_logs = []

    chat_logs.append("System message for LLM1: " + instr_llm1)
    chat_logs.append("System message for LLM2: " + instr_llm2)

    start = time.time()

    print("System message for LLM1: ", instr_llm1)
    print("System message for LLM2: ", instr_llm2)

    # Negotiation loop
    for turn in range(max_turns):
        turn_msg = ""
        if turn < max_turns - 1:
            turn_msg=f"Round {turn+1}:"
        else:
            turn_msg=f"Final round:"
        
        # if "deepseek" in cfg["model_name"]:
        #     if type(llm1_messages[-1]) == HumanMessage:
        #         llm1_messages[-1].content += "\n" + turn_msg
        #     else:
        #         llm1_messages.append(HumanMessage(content=turn_msg))
        #     if type(llm2_messages[-1]) == HumanMessage:
        #         llm2_messages[-1].content += "\n" + turn_msg
        #     else:
        #         llm2_messages.append(HumanMessage(content=turn_msg))
        # else:
        llm1_messages.append(HumanMessage(content=turn_msg))
        llm2_messages.append(HumanMessage(content=turn_msg))

        chat_logs.append("###################")
        chat_logs.append(f"Turn {turn+1}:")

        curr_hist = {}
        cur_turn = turn + 1
        print(f"Turn {cur_turn}")

        curr_hist["turn"] = cur_turn

        # LLM1 ToM reasoning

        # if "deepseek" in cfg["model_name"]:
        #     if type(llm1_messages[-1]) == HumanMessage:
        #         llm1_messages[-1].content += "\n" + proposer_tom
        # else:

        if cfg['tom_type'] in ['cot', 'zero', 'first', 'both']:
            llm1_messages.append(HumanMessage(content=proposer_tom))
            
            llm1_tom_response = llm1.invoke(input=llm1_messages)
            llm1_tom = llm1_tom_response.content
            llm1_messages.append(AIMessage(content=llm1_tom))
            print("LLM1 ToM: ", llm1_tom)

            chat_logs.append(f"Player {proposer_name} ToM reasoning: " + llm1_tom)
            curr_hist["proposer_tom"] = llm1_tom

        # LLM1 decision + strategic reasoning
        llm1_messages.append(HumanMessage(content=prop_decision_prompt))
        llm1_response = llm1.invoke(input=llm1_messages)
        llm1_prop_content = llm1_response.content

        print("LLM1 proposal: ", llm1_prop_content)

        # LLM1 save responses
        llm1_messages.append(AIMessage(content=llm1_prop_content))

        llm1_prop_content = llm1_prop_content.replace('Final Proposal: ', '')
        llm1_prop_content = llm1_prop_content.replace(f'**Round {turn+1} Proposal:**', '')

        if '<think>' in llm1_prop_content:
            llm1_prop = llm1_prop_content.split('</think>')[-1].strip()
            try:
                llm1_strat = "Strategy: " + llm1_prop.split('|')[1].strip().split('Strategy: ')[-1]
                llm1_prop = "Decision: " + llm1_prop.split('|')[0].strip().split('Decision: ')[-1]
            except IndexError:
                llm1_strat = "Strategy: " + llm1_prop.split('\n')[1].strip().split('Strategy: ')[-1]
                llm1_prop = "Decision: " + llm1_prop.split('\n')[0].strip().split('Decision: ')[-1]
        else:
            try:
                llm1_prop = "Decision: " + llm1_prop_content.split("|")[0].strip().split('Decision: ')[-1]
                llm1_strat = "Strategy: " + llm1_prop_content.split("|")[1].strip().split('Strategy: ')[-1]
            except IndexError:
                llm1_prop = "Decision: " + llm1_prop_content.split("\n")[0].strip().split('Decision: ')[-1]
                llm1_strat = "Strategy: " + llm1_prop_content.split("\n")[1].strip().split('Strategy: ')[-1]

        # if "deepseek" in cfg["model_name"]:
        #     if type(llm2_messages[-1]) == HumanMessage:
        #         llm2_messages[-1].content += "\n" + llm1_prop
        # else:
        llm2_messages.append(HumanMessage(content=llm1_prop))

        chat_logs.append(f"Player {proposer_name} proposal: " + llm1_prop)
        curr_hist["proposer"] = llm1_prop

        chat_logs.append(f"Player {proposer_name} strategic reasoning: " + llm1_strat)
        curr_hist["proposer_strat"] = llm1_strat

        #######################

        # LLM2 ToM reasoning
        # if "deepseek" in cfg["model_name"]:
        #     if type(llm2_messages[-1]) == HumanMessage:
        #         llm2_messages[-1].content += "\n" + responder_tom
        # else:

        if cfg['tom_type'] in ['cot', 'zero', 'first', 'both']:
            llm2_messages.append(HumanMessage(content=responder_tom))

            llm2_tom_response = llm2.invoke(input=llm2_messages)
            llm2_tom = llm2_tom_response.content
            llm2_messages.append(AIMessage(content=llm2_tom))
            print("LLM2 ToM: ", llm2_tom)
            
            chat_logs.append(f"Player {responder_name} ToM reasoning: " + llm2_tom)
            curr_hist["responder_tom"] = llm2_tom

        # LLM2 decision + strategic reasoning
        llm2_messages.append(HumanMessage(content=resp_decision_prompt))
        llm2_response = llm1.invoke(input=llm2_messages)
        llm2_prop_content = llm2_response.content
        print("LLM2 decision: ", llm2_prop_content)

        if '<think>' in llm2_prop_content:
            llm2_prop = llm2_prop_content.split('</think>')[-1].strip()
            try:
                llm2_strat = "Strategy: " + llm2_prop.split('|')[1].strip().split('Strategy: ')[-1]
                llm2_prop = "Decision: " + llm2_prop.split('|')[0].strip().split('Decision: ')[-1]
            except IndexError:
                llm2_strat = "Strategy: " + llm2_prop.split('\n')[1].strip().split('Strategy: ')[-1]
                llm2_prop = "Decision: " + llm2_prop.split('\n')[0].strip().split('Decision: ')[-1]
        else:
            try:
                llm2_prop = "Decision: " + llm2_prop_content.split("|")[0].strip().split('Decision: ')[-1]
                llm2_strat = "Strategy: " + llm2_prop_content.split("|")[1].strip().split('Strategy: ')[-1]
            except IndexError:
                llm2_prop = "Decision: " + llm2_prop_content.split("\n")[0].strip().split('Decision: ')[-1]
                llm2_strat = "Strategy: " + llm2_prop_content.split("\n")[1].strip().split('Strategy: ')[-1]

        # LLM2 save responses
        llm2_messages.append(AIMessage(content=llm2_prop_content))

        # if "deepseek" in cfg["model_name"]:
        #     if type(llm1_messages[-1]) == HumanMessage:
        #         llm1_messages[-1].content += "\n" + llm2_prop
        # else:
        llm1_messages.append(HumanMessage(content=llm2_prop))
        # llm1_messages.append(HumanMessage(content=llm2_prop))

        chat_logs.append(f"Player {responder_name} decision: " + llm2_prop)
        curr_hist["responder"] = llm2_prop

        chat_logs.append(f"Player {responder_name} strategic reasoning: " + llm2_strat)
        curr_hist["responder_strat"] = llm2_strat

        #######################

        if cfg['post-response_reflection']:
            llm1_messages.append(HumanMessage(content=proposer_expl))
            llm1_expl_response = llm1.invoke(input=llm1_messages)
            llm1_expl = llm1_expl_response.content
            llm1_messages.append(AIMessage(content=llm1_expl))
            print("LLM1 explanation: ", llm1_expl)

            llm2_messages.append(HumanMessage(content=responder_expl))
            llm2_expl_response = llm2.invoke(input=llm2_messages)
            llm2_expl = llm2_expl_response.content
            llm2_messages.append(AIMessage(content=llm2_expl))
            print("LLM2 explanation: ", llm2_expl)
        else:
            llm1_expl = ""
            llm2_expl = ""
        
        chat_logs.append(f"Player {proposer_name} proposal explanation: " + llm1_expl)
        curr_hist["proposer_expl"] = llm1_expl

        chat_logs.append(f"Player {responder_name} response explanation: " + llm2_expl)
        curr_hist["responder_expl"] = llm2_expl

        dialogue_hist.append(curr_hist)

        if "accept" in llm2_prop.lower():
            print("B has accepted the offer. Game will end.")
            chat_logs.append("B has accepted the offer. Game will end.")
            break
        elif "reject" in llm2_prop.lower():
            print("B has rejected the offer and the game will continue if not at the maximum number of rounds.")
            chat_logs.append("B has rejected the offer and the game will continue if not at the maximum number of rounds.")

    end = time.time()
    print(f"End in {end-start:.2f} sec.")

    return (dialogue_hist, chat_logs)

if __name__ == "__main__":
    simulations()