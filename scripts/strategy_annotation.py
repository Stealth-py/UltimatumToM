from argparse import ArgumentParser
import openai
import pandas as pd
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

class Annot(BaseModel):
    question1: str = Field(enum=['yes', 'no'])
    question2: str = Field(enum=['yes', 'no'])

P_ANNOTATION_PROMPT = """
You are a skilled annotator who can measure the consistency between two free texts.
You will be provided with a proposal from a player A in an Ultimatum Game, their intended beliefs and their reasoning behind the proposal.
You need to verify if the reasoning is consistent with the proposal of player A and separately if the reasoning is consistent with intended beliefs of player A. For example, the player can be fair, greedy or selfless, so the reasoning should be reflective of this belief.


Proposal: [@prop]
Belief: [@belief]
Reasoning: [@reasoning]


Verify the consistencies between reasoning - proposal (question1) and reasoning - belief (question2).
"""

R_ANNOTATION_PROMPT = """
You are a skilled annotator who can measure the consistency between two free texts.
You will be provided with a proposal from a player A in an Ultimatum Game, the decision a player B makes, intended beliefs of player B, and reasoning of player B behind making the decision.
You need to verify if the reasoning is consistent with the decision of player B and separately if the reasoning is consistent with the intended beliefs of player B. For example, the player can be fair, greedy or selfless, so the reasoning should be reflective of this belief.


Proposal: [@prop]
Decision: [@decision]
Belief: [@belief]
Reasoning: [@reasoning]


Verify the consistencies between reasoning - decision (question1) and reasoning - belief (question2).
"""

def get_response(prop, belief, reasoning, decision=''):
    if decision:
        prompt = R_ANNOTATION_PROMPT.replace('[@prop]', prop).replace('[@decision]', decision).replace('[@belief]', belief).replace('[@reasoning]', reasoning)
    else:
        prompt = P_ANNOTATION_PROMPT.replace('[@prop]', prop).replace('[@belief]', belief).replace('[@reasoning]', reasoning)

    resp = client.responses.parse(
        model="gpt-4o",
        input=[
            {'role': 'user', 'content': prompt}
        ],
        text_format=Annot,
        temperature=0,
    ).output_parsed

    return resp.question1, resp.question2

if __name__ == "__main__":
    for file in os.listdir('human_verification'):
        print(file)
        q1 = []
        q2 = []
        
        if '.csv' not in file:
            continue

        if 'prop' in file:
            df = pd.read_csv(os.path.join('human_verification', file))
            # del df['gpt-4o']

            prop = df['Proposal'].values.tolist()
            belief = df['Proposer Belief'].values.tolist()
            reason = df['Proposer Reasoning'].values.tolist()

            for i in tqdm(range(len(df))):
                x = get_response(prop[i], belief[i], reason[i])
                q1.append(x[0])
                q2.append(x[1])
            
            df['q1'] = q1
            df['q2'] = q2

            df.to_csv(os.path.join('human_verification', file), index=False)
        
        elif 'resp' in file:
            df = pd.read_csv(os.path.join('human_verification', file))

            prop = df['Proposal'].values.tolist()
            decision = df['Decision'].values.tolist()
            belief = df['Responder Belief'].values.tolist()
            reason = df['Responder Reasoning'].values.tolist()

            for i in tqdm(range(len(df))):
                x = get_response(prop[i], belief[i], reason[i], decision=decision[i])
                q1.append(x[0])
                q2.append(x[1])
            
            df['q1'] = q1
            df['q2'] = q2

            df.to_csv(os.path.join('human_verification', file), index=False)