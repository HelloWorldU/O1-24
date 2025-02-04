import heapq
from typing import List, Tuple, Optional
import math
import re
import itertools
import numpy as np
from fractions import Fraction
from src.evaluator import Evaluator
from functools import partial
from collections import defaultdict

EVAL_MAP = {
    "impossible": 0.1,
    "likely": 1.0,
    "sure": 10.0
}

class State:
    def __init__(
        self,
        q: str,
        numbers: List[float],
        steps: int = 0,
        path: Optional[List[str]] = None,
        eval_history: Optional[List[float]] = None,
    ):
        self.q = q
        self.numbers = numbers
        self.steps = steps
        self.path = path if path is not None else []
        self.eval_history = eval_history if eval_history is not None else []

    def state_hash(self) -> str:
        question = ','.join(f"{n}" for n in self.numbers)
        path_key = '->'.join(self.path)
        return f"{question}||{path_key}"


class AStarSolver:
    def __init__(self, max_depth: int=3, lambda_decay: float=0.8, k: int=12):
        self.max_depth = max_depth
        self.lambda_decay = lambda_decay
        self.k = k

    def _evaluate_step(self, task, q: str, y: str) -> Tuple[List[str], float]:
        content = get_evaluations(task, q, y).split('\n')
        last_line = content[-1]
        eval_val = None
        for key in EVAL_MAP:
            if key in last_line:
                eval_val = EVAL_MAP[key]
                break
        if eval_val is None:
            eval_val = EVAL_MAP["impossible"]
        return content, eval_val

    def _calculate_priority(self, state: State) -> float:
        g = state.steps
        n = len(state.eval_history)
        v_t = sum(self.lambda_decay ** (n - i - 1) * eval_val for i, eval_val in enumerate(state.eval_history))
        if state.steps > 1:
            v_max = sum(self.lambda_decay ** (state.steps - i - 1) * EVAL_MAP["likely"] for i in range(state.steps - 1))
            v_max += self.lambda_decay * EVAL_MAP["sure"]
        else:
            v_max = self.lambda_decay * EVAL_MAP["sure"]
        h = 1 - (v_t / v_max)
        
        return g + h

    def _generate_operations(self, task, q: str, nums: List[float]) -> Tuple[List, List]:
        return get_proposals(task, q, nums, self.k)
    
    def solve(self, task, q: str, input_numbers: List[float]) -> List[str]:
        initial_state = State(q, input_numbers)
        initial_priority = self._calculate_priority(initial_state)
        heap = [(initial_priority, initial_state)]
        visited = set()
        answer = []

        while heap:
            current_priority, current_state = heapq.heappop(heap)
            answer.append('\n'.join(current_state.path))

            if len(current_state.numbers) == 1:
                if math.isclose(current_state.numbers[0], 24, rel_tol=1e-4):
                    return answer

            if current_state.steps == self.max_depth:
                continue

            proposals = self._generate_operations(task, q, current_state.numbers)
            for new_nums, step_desc in proposals:
                infer_path, new_eval = self._evaluate_step(task, new_nums, step_desc)

                new_state = State(
                    q=q,
                    numbers=new_nums,
                    steps=current_state.steps + 1,
                    path=current_state.path + infer_path,
                    eval_history=current_state.eval_history + [new_eval]
                )
                state_key = new_state.state_hash()
                if state_key in visited:
                    continue
                visited.add(state_key)

                new_priority = self._calculate_priority(new_state)
                heapq.heappush(heap, (new_priority, new_state))

        return answer


def get_proposals(task, q, y, k): 
    propose_prompt = task.propose_prompt_wrap(q, y, k)
    proposals = evaluator.generate_response(propose_prompt)
    return format_prompt(proposals)


def get_evaluations(task, q, y):
    value_prompt = task.value_prompt_wrap(q, y)
    evaluations = evaluator.generate_response(value_prompt)
    return evaluations


# get numbers after proposal prompt
def format_prompt(prompt):
    new_proposals = [_ for _ in prompt]
    deal_num = []
    left_num = []
    for i, step in enumerate(new_proposals):
        deal_num.append(step.split('(')[0])
        left_num.append(step.split('(')[1].split(')')[0])
    return (deal_num, left_num)


def a_star_solve(args, task, idx, k, to_print=True):
    global evaluator
    evaluator = Evaluator(backend=args.backend, temperature=args.temperature)
    q = task.get_input(idx)  # question
    ys = ['']  # current output candidates
    AStar = AStarSolver(max_depth=task.steps, k=k)
    answer_list = AStar.solve(task, q, list(map(float, re.findall(r'\d+', q))))
    
    if to_print:
        print(ys)
    build_dataset(answer_list, idx)


def build_dataset(answer_list, idx):
    # build dataset
    pass