import re
import os
import sympy
import random
import pandas as pd
from src.base import Task, DATA_PATH, MODEL_PATH, model_path
from src.prompts import * 
from typing import List, Dict, Tuple


class Game24Task(Task):
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.df = pd.read_csv(path)
        # self.sort_data()  # It's truly annoying
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.steps = 3

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def sort_data(self) -> None:
        df = self.df.copy()
        df["Solved rate"] = df["Solved rate"].astype(str).str.rstrip('%')
        df["Solved rate"] = pd.to_numeric(df["Solved rate"], errors="coerce")
        df = df.sort_values("Solved rate", ascending=False)

        self.df = df.copy()
        updated_path = os.path.join(DATA_PATH, '24', '24.csv')
        self.df.to_csv(updated_path, index=False)

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
    def value_prompt_wrap(x: str, y: str) -> str:
        prompt = value_prompt.format(input=y)
        # print(prompt)
        return prompt
    
    @staticmethod
    def value_last_step_wrap(x: str, y: str) -> str:
        prompt = value_last_step_prompt.format(input=x, expression=y)
        # print(prompt)
        return prompt
    
    @staticmethod
    def combine_prompt_wrap(x: str, y: str) -> str:
        prompt = combine_prompt.format(input=x, operations=y)
        # print(prompt)
        return prompt
    
    @staticmethod
    def propose_outputs(q: str, numbers: List[float]) -> Tuple[List[str], List[str]]:
        visit_op = set()
        deal_num, left_num = [], []
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                x, y = numbers[i], numbers[j]
                remaining = [numbers[k] for k in range(len(numbers)) if k not in (i, j)]
                candidate_ops = [
                    (x + y, f"{x} + {y}"),       
                    (x * y, f"{x} * {y}"),     
                    (x - y, f"{x} - {y}"),    
                    (y - x, f"{y} - {x}")        
                ]
                if y != 0:
                    candidate_ops.append((x / y, f"{x} / {y}"))
                if x != 0:
                    candidate_ops.append((y / x, f"{y} / {x}"))
                
                for result, op_str in candidate_ops:
                    if op_str in visit_op:
                        continue
                    visit_op.add(op_str)
                    deal_num.append(f"{op_str} = {result}")
                    new_remaining = sorted(remaining + [result])
                    left_num.append(" ".join(str(num) for num in new_remaining))

        return deal_num, left_num