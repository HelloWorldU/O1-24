import heapq
from typing import List, Tuple, Dict
import math
import re
from src.evaluator import Evaluator
import logging
import sys
import sympy

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     stream=sys.stdout,
#     force=True
# )
# logger = logging.getLogger(__name__)
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
        path: List[Tuple[str, List[str]]] = None,
        eval_history: List[float] = None,
    ):
        """
        Parameters:
            q (str):
                The question string.
            numbers (List[float]):
                The current list of numbers.
            steps (int):
                The number of steps taken so far.
            path (List[Tuple[str, List[str]]]):
                A list of tuples where each tuple represents one step in the reasoning path.
                The first element of the tuple is the operation description (a string), and the
                second element is a list of evaluation messages for that operation.
                Example: 
                    [("1 + 1 = 2", [".....Likely!"])]
                    would become [("1 + 1 = 2", [".....Likely!"]), ("2 + 20 = 24", [".....Sure!"])]
            eval_history (List[float]):
                A list of evaluation scores for each step.
        """
        self.q = q
        self.numbers = numbers
        self.steps = steps
        self.path = path if path else []
        self.eval_history = eval_history if eval_history else []

    def __lt__(self, other):
        return self.steps < other.steps
    
    def state_hash(self) -> str:
        paths = [path[0] for path in self.path]
        path_key = ' -> '.join(paths)
        return f"{path_key}"


class AStarSolver:
    def __init__(self, max_depth: int=3, lambda_decay: float=0.8):
        self.max_depth = max_depth
        self.lambda_decay = lambda_decay

    def _evaluate_step(self, task, q: str, y: str) -> Tuple[List[str], float]:
        #   TODO: repeat sampling and majority vote
        evaluations = get_evaluations(task, q, y)
        # logger.info("Evaluations: %s", evaluations)
        if evaluations is None:
            return ["Impossible."], EVAL_MAP["impossible"]
        content = evaluations.split('\n')
        pattern = re.compile(r'\\boxed\{\\text\{(.*)\}\}')
        eval = re.findall(pattern, evaluations)[-1].lower()
        eval_val = None
        for key in EVAL_MAP:
            if key == eval:
                # logger.info("Matched eval: %s", eval)
                eval_val = EVAL_MAP[key]
                # logger.info("Eval value: %s", eval_val)
                break
        content = content[:-1] if "boxed" in content[-1] else content[:-5]
        # logger.info("Last line: %s", content[-1])
        return content, eval_val if eval_val else EVAL_MAP["impossible"]
    
    def _evaluate_last_step(self, x: str, y: str) -> Tuple[List[str], float]:
        if math.isclose(float(y), 24, abs_tol=1e-4):
            return [f"{x}.\nSure!"], EVAL_MAP["nSure"]
        else:
            return [f"{x}.\nImpossible!"], EVAL_MAP["impossible"]

    def _calculate_priority(self, state: State) -> float:
        g = state.steps
        v_t = sum(self.lambda_decay ** (i + 1) * eval_val for i, eval_val in enumerate(state.eval_history))
        if state.steps > 1:
            v_max = sum(self.lambda_decay ** (state.steps - i - 1) * EVAL_MAP["likely"] for i in range(state.steps - 1))
            v_max += self.lambda_decay ** (state.steps) * EVAL_MAP["sure"]
        else:
            v_max = self.lambda_decay * EVAL_MAP["sure"]
        h = 1 - (v_t / v_max)
        
        return g + h

    def _generate_operations(self, task, q: str, nums: List[float]) -> Tuple[List[str], List[str]]:
        return get_proposals(task, q, nums)
    
    def solve(self, task, q: str, input_numbers: List[float]) -> List[str]:
        initial_state = State(q, input_numbers)
        initial_priority = self._calculate_priority(initial_state)
        heap = [(initial_priority, initial_state)]
        visited = set()
        answer = ["|<begin_of_thought>|"]
        while heap:
            current_priority, current_state = heapq.heappop(heap)

            if current_state.steps == self.max_depth:
                if math.isclose(sympy.sympify(current_state.numbers[0]), 24, abs_tol=1e-4):
                    one_solution = format_path(task, current_state.q, current_state.path)
                    final_answer = format_final_answer(one_solution)
                    # logger.info("Final answer: %s", final_answer)
                    if "|<end_of_thought>|" not in answer:
                        answer.append("|<end_of_thought>|")
                    answer.append(final_answer)
                    return answer
                else:
                    continue    # Skip this state if it has reached the maximum depth and the target number is not 24.

            if current_state.path:
                current_path = immediate_attempts(current_state.path)
                answer.append(format_eval_path(current_path))
            
            # logger.info("Current question: %s", current_state.q)
            new_numbers, left_numbers = self._generate_operations(task, current_state.q, current_state.numbers)
            for new_nums, step_desc in zip(new_numbers, left_numbers):
                # logger.info("New numbers: %s", new_nums)
                # logger.info("Left numbers: %s", step_desc)
                numbers_in_step = re.findall(r'\d+\.?\d*', step_desc)
                if len(numbers_in_step) == 1:
                    infer_path, new_eval = self._evaluate_last_step(new_nums, step_desc)
                else:
                    infer_path, new_eval = self._evaluate_step(task, current_state.q, step_desc)

                new_path = (new_nums, infer_path)
                merged_path = current_state.path + [new_path]
                current_num = list(map(float, re.findall(r'\d+\.?\d*', step_desc)))

                new_state = State(
                    q=current_state.q,
                    numbers=current_num,
                    steps=current_state.steps + 1,
                    path=merged_path,
                    eval_history=current_state.eval_history + [new_eval]
                )

                # early pruning
                if new_eval == EVAL_MAP["sure"] and len(numbers_in_step) > 1:
                    expression, flag = determinant_solution(task, new_state)
                    if flag:
                        # logger.info("Early pruning")
                        one_solution = format_path(task, new_state.q, new_state.path, expression)
                        # logger.info("One solution: %s", one_solution)
                        final_answer = format_final_answer(one_solution)
                        if "|<end_of_thought>|" not in answer:
                            paths = ["Try " + op + "." + ''.join(path) for op, path in new_state.path]
                            answer.append(f"{chr(10).join(paths)}\n|<end_of_thought>|")
                        answer.append(final_answer)
                        return answer
                
                state_key = new_state.state_hash()
                if state_key in visited:
                    continue
                visited.add(state_key)

                new_priority = self._calculate_priority(new_state)
                heapq.heappush(heap, (new_priority, new_state))
            
        return answer


def determinant_solution(task, state) -> Tuple[str, bool]:
    infer_path = '\n'.join(state.path[-1][1])
    # pattern = re.compile(r'\\(?:\(|\[)(?!\\boxed{)\s*(.*)(?:\\\)|\\\])')
    patterns = [
        re.compile(r'\((.*?)\s*=\s*24\s*\)'),
        re.compile(r'\\\(\s*(.*?)\s*=\s*24\s*\\\)'),
        re.compile(r'\\\[\s*(.*?)\s*=\s*24\s*\\\]'),
        re.compile(r'\((.*?)\s*=\s*24.0\s*\)'),
        re.compile(r'\\\(\s*(.*?)\s*=\s*24.0\s*\\\)'),
        re.compile(r'\\\[\s*(.*?)\s*=\s*24.0\s*\\\]'),
    ]

    for pattern in patterns:
        matches = re.findall(pattern, infer_path)
        if matches:
            break
    # logger.info("matches: %s", matches)
    expression = f"{matches[-1]} = 24"
    # logger.info("expression: %s", expression)
    last_eval_msg = f"{state.path[-1][0]}\n{expression}"
    prompt = task.value_last_step_wrap(state.q, last_eval_msg)
    flag = evaluator.generate_response(prompt)
    # logger.info("flag: %s", flag)
    boxed = '\n'.join(flag.split('\n')[-5:]).lower()
    # logger.info("boxed: %s", boxed)
    return (expression, True) if "sure" in boxed else (expression, False)


def get_proposals(task, q, y) -> Tuple[List[str], List[str]]:
    return task.propose_outputs(q, y)


def get_evaluations(task, q, y) -> str:
    value_prompt = task.value_prompt_wrap(q, y)
    evaluations = evaluator.generate_response(value_prompt)
    return evaluations


def format_final_answer(solution: str) -> str:
    final_answer = "|<begin_of_solution>|\n" + solution + "\n|<end_of_solution>|"
    return final_answer


def format_eval_path(path_values: List[str]) -> str:
    return "\n".join(path_values)


def format_path(task, q: str, path: List[Tuple[str, List[str]]], expression: str=None) -> List[str]:
    operations = [op for op, _ in path]
    operations = ', '.join(operations)
    operations += ", "+ expression if expression else ""
    # logger.info("Operations: %s", operations)
    prompt = task.combine_prompt_wrap(q, operations)
    return evaluator.generate_response(prompt)


def immediate_attempts(path) -> List[str]:
    attempts = ["Try "+ op + "." + '\n'.join(desc) for op, desc in path]
    return ''.join(attempts).split('\n')


def a_star_solve(args, task, idx):
    global evaluator
    evaluator = Evaluator(backend=args.backend, temperature=args.temperature)
    q = task.get_input(idx)  # question
    AStar = AStarSolver(max_depth=task.steps)
    answer_list = AStar.solve(task, q, list(map(float, re.findall(r'\d+\.?\d*', q))))
    return answer_list
