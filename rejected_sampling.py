from heapq import heappush, heappop
import io
from typing import List, Tuple, Set, Dict
from contextlib import redirect_stdout
import math
import random


def generate_enhanced_solution(numbers: List[int], solve_24_with_steps, tokenizer) -> Dict:
    output = io.StringIO()
    with redirect_stdout(output):
        solution_expression = solve_24_with_steps(numbers)
    
    output_text = output.getvalue()
    output_text = optimize_output(output_text)
    correct_path = []
    flag = False
    explanation_Yes = ""
    explanation_No = ""

    for line in output_text.split('\n'):
        if not flag:
            if line.startswith('Find the solution!'):
                flag = True
            else:
                continue
        elif flag:
            if line.startswith('Final expression:'):
                break
            else:
                correct_path.append(line)

    if correct_path:
        explanation_Yes = (
            # f"Let me explain how we found this solution:\n\n"
            # f"1. Initial Setup:\n"
            # f"   - Starting numbers: {numbers}\n"
            # f"   - Target value: 24\n"
            # f"   - Available operations: +, -, *, /\n"
            # f"   - Each number must be used exactly once\n\n"
            f"2. Search Strategy:\n"
            f"   - Implemented A* search algorithm\n"
            f"   - Used heuristic function h(n) = |24 - current_value|\n"
            f"   - Maintained priority queue for optimal path exploration\n"
            f"   - Tracked visited states to prevent cycles\n\n"
            f"3. Solution Path:\n"
            + "\n".join(f"   {step}" for step in correct_path) +
            f"\n\n4. Final Result:\n"
            f"   We successfully found that {solution_expression} = 24\n"
            f"   This solution is optimal as it uses the minimum necessary operations."
        )
    else:
        explanation_No = (
            f"\n\nFinal Result:\n"
            f"   No solution found.\n"
            f"   This indicate that the problem is unsolvable or that the problem is too complex for the model to solve."
        )    

    # Construct the enhanced solution dictionary
    thoughts = (
#             f"So, I've got this question：Given 4 numbers {numbers}, using add, subtract, multiply, and divide (each number can only be used once\
# , and parentheses can be used to change the order of operations) so that the final result is 24."
#             f"Let me think carefully, the rule is We can only use basic arithmetic operations (+, -, *, /)."
#             f"And it seems like that we can't modify individual numbers (like making them negative directly)."
#             f"Each number must be used exactly once."
#             f"The final result must be exactly 24 (not -24)."
#             f"Ok, l should be careful, let me solve this 24 Game problem using the A* search algorithm. "
#             f"\n\nInitial numbers: {numbers}"
#             f"\n\nThe A* algorithm will help us find the optimal solution by:"
#             f"\n1. Using a heuristic function to estimate how close we are to 24"
#             f"\n2. Maintaining a priority queue to explore most promising paths first"
#             f"\n3. Keeping track of visited states to avoid cycles"
#             f"\n\nSearch Process:"
            f"\n{output_text}"
        )
    return {
        "question": (
            f"Given the numbers {numbers}, use arithmetic operations (+, -, *, /) "
            "and parentheses to create an expression that equals 24. "
            "Each number must be used exactly once."
        ),
        
        "thought_process": thoughts,
        
        "solution": f"\\[ \\boxed{{{solution_expression} = 24}} \\]",
        
        "explanation": explanation_Yes if correct_path else explanation_No,
        "tokens": len(tokenizer(thoughts)['input_ids'])
    }


def optimize_output(output: str, num_samples: int=5):
    lines = output.split('\n')
    blocks = []
    current_tries = []
    try_cnt = 0
    result_section = []
    flag = False
    for line in lines:
        if line.startswith('Try:'):
            try_cnt += 1
        else:
            break
    
    for line in lines[try_cnt:]:
        if line.startswith('Chose:') and not flag:
            current_tries.append(line)
        elif line.startswith('Try:') and not flag:
            if current_tries:
                blocks.append(current_tries + [line])
                current_tries = []
        else:
            result_section.append(line)
            flag = True

    selected_blocks = []
    if blocks:
        num_samples = random.randint(num_samples, 10)
        num_samples = min(num_samples, len(blocks))
        indices = sorted(random.sample(range(len(blocks)), num_samples))
        selected_blocks = [blocks[i] for i in indices]

    optimized_lines = []
    for block in selected_blocks:
        optimized_lines.extend(block)
    
    optimized_lines.extend(result_section)
    
    final_output = '\n'.join(optimized_lines)
    
    return final_output