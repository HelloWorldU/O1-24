value_prompt = '''Evaluate if given numbers can obtain 24 (sure/likely/impossible)
Attention:
Only use the input numbers to obtain 24, using basic arithmetic operations (+ - * /).
All numbers must be used exactly once.
Use as little combinations as necessary. When you find a valid way to get 24, stop further exploration.
For each attempted combination, provide natural language reasoning (explain your thought process step-by-step) before deciding whether to continue or conclude.
Do not enumerate every possible combination; only show the attempts that lead you toward the solution.
If you find a way to get 24, output the complete combination in one expression on the last second line using the exact format \(6 + 6 + 12 = 24\) where all numbers and operations are enclosed in ().
On the last line, output the final evaluation result exactly in the format \boxed{{\text{{eval value}}}}. Do not output the boxed answer more than once. The braces must be curly brackets.
If you can obtain 24, the eval value should be sure; if you can't, the eval value should be impossible, if you are not sure, the eval value should be likely. Do not use other eval value.
Input: {input}
'''


value_last_step_prompt = """Following input are consist of four numbers and some operations, four numbers stand the initial input of the question of using basic arithmetic operations (+ - * /) to obtain 24, and operations are mid attempts step to obtain 24,judge the expression could obtain 24(Sure/Wrong)
Attention:
Please compute both sides of the math expression in the giving expression. If they are all equal, mark this condition is 'Sure!'; otherwise, 'Wrong!'.
Also check whether the numbers in expression were come from the result of the operations in input or not. If they are correct to combine to obtain 24, mark this condition is 'Sure!'; otherwise, 'Wrong!'.
Two conditions are all satisfied, output 'Sure!'; otherwise, output 'Wrong!'.
Input: {input}
Expression: {expression}
Judge:
"""


combine_prompt = """The following input consists of four numbers and a series of operations.
These four numbers are the initial input for the problem of obtaining 24 using basic arithmetic operations (+, -, *, /).
The provided operations represent intermediate steps attempted in the process.
Your task is to combine these operations into one complete expression that uses exactly the four numbers provided as input, and each number should be used only once.

Attention:
- The final expression must use exactly all four numbers from the input list.
- No additional numbers or operations should be introduced, only use the four numbers in the exact input list.
- Each number must be used exactly once.
- Do not generate any additional intermediate steps—only output the single, complete expression.
- The final answer must be formatted exactly as: \\boxed{{expression}}

Input: {input}
Operations: {operations}
Combine:"""