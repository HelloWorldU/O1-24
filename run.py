"""
build dataset
"""

import os
import json
import argparse

from src.algorithm import AStarSolver, naive_solve, a_star_solve
from src.base import get_task
from src.evaluator import deepseek_usage

def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    
    # 文件路径定义
    if args.naive_run:
        log_file = f'./logs/{args.task}/{args.backend}{args.temperature}naive{args.prompt_sample}sample{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        log_file = f'./logs/{args.task}/{args.backend}{args.temperature}_{args.method_generate}{args.n_generate_sample}{args.method_evaluate}{args.n_evaluate_sample}{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    train_list, test_list = task._sample_data(args.task_start_index, args.task_end_index)
    for i in (train_list, test_list):
        # 解决任务
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info = a_star_solve(args, task, i, args.k)

        # 记录日志信息
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': deepseek_usage(args.backend)})
        logs.append(info)

        # 记录主要的指标
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')

    # 输出平均结果
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', deepseek_usage(args.backend))

    # 保存日志
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['deepseek-chat', 'gpt-4'], default='deepseek-chat')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--task', type=str, required=True, choices=['game24'])
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=1362)
    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--k proposals', type=int, default=12)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run
    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)