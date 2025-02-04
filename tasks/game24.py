import re
import os
import sympy
import random
import pandas as pd
from src.base import Task, DATA_PATH, MODEL_PATH, model_path
from src.prompts import * 
from typing import List, Dict, Tuple

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.df = pd.read_csv(path)
        self.sort_data()
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 3
        self.stops = ['\n'] * 3

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str) -> dict:
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}

    def sort_data(self) -> None:
        df = self.df.copy()
        df["Solved rate"] = df["Solved rate"].astype(str).str.rstrip('%')
        df["Solved rate"] = pd.to_numeric(df["Solved rate"], errors="coerce")
        df = df.sort_values("Solved rate", ascending=False)

        self.df = df.copy()
        updated_path = os.path.join(DATA_PATH, '24', '24.csv')
        self.df.to_csv(updated_path, index=False)
        # print(f"Update success: {updated_path}")

    def _sample_data(self, start_idx: int, end_idx: int, train_size: int = 300, test_size: int = 100) -> Tuple[List, List]:
        df = self.df.copy()
        df = df.iloc[start_idx:end_idx]
        
        groups = df.groupby("difficulty")
        train_frames, test_frames = [], []
        train_set, test_set = set(), set()
        total = len(df)
        
        for _, group in groups:
            prop = len(group) / total
            n_train = int(round(train_size * prop))
            n_test  = int(round(test_size * prop))
            group_train = group.sample(n=n_train, random_state=42)
            group_test  = group.drop(group_train.index).sample(n=n_test, random_state=42)
            train_frames.append(group_train)
            test_frames.append(group_test)
            train_set.update(group_train.index)
            test_set.update(group_test.index)

        current_train = sum(len(frame) for frame in train_frames)
        if current_train < train_size:
            extra = df.loc[~df.index.isin(train_set | test_set)].sample(n=train_size - current_train, random_state=42)
            train_frames.append(extra)
        
        current_test = sum(len(frame) for frame in test_frames)
        if current_test < test_size:
            extra = df.loc[~df.index.isin(train_set | test_set)].sample(n=test_size - current_test, random_state=42)
            test_frames.append(extra)
        
        train_df = pd.concat(train_frames).sort_index()
        test_df  = pd.concat(test_frames).sort_index()
        
        return train_df.index.tolist(), test_df.index.tolist()
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='', k: int=12) -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(k=k, input=current_numbers)
            print(prompt)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        # last_line = y.strip().split('\n')[-1]
        # if 'left: ' not in last_line:  # last step
        #     ans = last_line.lower().replace('answer: ', '')
        #     # print([value_last_step_prompt.format(input=x, answer=ans)])
        #     return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.1, 'likely': 1, 'sure': 10}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value