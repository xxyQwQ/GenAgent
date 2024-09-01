import os
import yaml
import json
import argparse
import pandas as pd

from utils.parser import parse_code_to_workflow, parse_markdown_to_workflow
from utils.comfy import execute_workflow


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    proxy_config = config['proxy']

os.environ['http_proxy'] = proxy_config['http_proxy']
os.environ['https_proxy'] = proxy_config['https_proxy']


def main(args):
    with open('./dataset/query/meta.json', 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)

    record = dict()
    for agent_name in args.agent_name:
        record[agent_name] = dict()
        for task_id, task_info in metadata.items():
            record[agent_name][f'task_{task_id}'] = {
                'task_info': task_info,
                'num_runs': 0,
                'num_passes_1': 0,
                'num_passes_2': 0
            }

    for agent_name in args.agent_name:
        print(f'[Evaluation] agent {agent_name}')

        for task_id in metadata.keys():
            prefix = f'{args.save_path}/{agent_name}/task_{task_id}'
            print(f'[Evaluation] task {task_id}')

            for run_id in os.listdir(prefix):
                record[agent_name][f'task_{task_id}']['num_runs'] += 1
                checkpoint = os.path.join(prefix, run_id)
                print(f'[Evaluation] checkpoint {checkpoint}')

                # Skip: already evaluated
                output_path = os.path.join(checkpoint, 'output')
                if os.path.exists(output_path):
                    record[agent_name][f'task_{task_id}']['num_passes_1'] += 1
                    record[agent_name][f'task_{task_id}']['num_passes_2'] += 1
                    print('skipped: already evaluated')
                    continue

                # Check: pipeline error
                log_path = os.path.join(checkpoint, 'run.log')
                with open(log_path, 'r', errors='ignore') as file:
                    log = file.read()
                if 'Failed to generate workflow' in log or 'Failed to refine workflow' in log:
                    print('skipped: pipeline error')
                    continue

                # Case: standard representation
                if agent_name in [
                    'zero_shot_agent',
                    'few_shot_agent',
                    'cot_agent',
                    'rag_agent',
                    'gen_agent',
                    'code_gen_agent',
                    'single_gen_agent'
                ]:
                    # Check: no file
                    code_path = os.path.join(checkpoint, 'code.py')
                    if not os.path.exists(code_path):
                        print('skipped: no file')
                        continue

                    # Check: empty code
                    with open(code_path, 'r') as file:
                        code = file.read()
                    if code.strip() == '':
                        print('skipped: empty code')
                        continue

                    # Check: invalid workflow
                    try:
                        workflow = parse_code_to_workflow(code)
                    except Exception as error:
                        print('skipped: invalid workflow')
                        continue

                # Case: json representation
                elif agent_name in [
                    'json_gen_agent'
                ]:
                    # Check: no file
                    json_path = os.path.join(checkpoint, 'workflow.json')
                    if not os.path.exists(json_path):
                        print('skipped: no file')
                        continue

                    # Check: invalid format
                    try:
                        with open(json_path, 'r') as file:
                            workflow = json.load(file)
                    except Exception as error:
                        print('skipped: invalid workflow')
                        continue

                # Case: list representation
                elif agent_name in [
                    'list_gen_agent',
                ]:
                    # Check: no file
                    markdown_path = os.path.join(checkpoint, 'markdown.md')
                    if not os.path.exists(markdown_path):
                        print('skipped: no file')
                        continue

                    # Check: empty list
                    with open(markdown_path, 'r') as file:
                        markdown = file.read()
                    if markdown.strip() == '':
                        print('skipped: empty list')
                        continue

                    # Check: invalid workflow
                    try:
                        workflow = parse_markdown_to_workflow(markdown)
                    except Exception as error:
                        print('skipped: invalid workflow')
                        continue

                # Record: pass 1
                record[agent_name][f'task_{task_id}']['num_passes_1'] += 1

                # Check: execution failure
                try:
                    status, outputs = execute_workflow(workflow)
                except Exception as error:
                    print('skipped: execution failure')
                    continue

                # Check: invalid status
                if status['status_str'] != 'success':
                    print('skipped: invalid status')
                    continue

                # Check: empty output
                if len(outputs) == 0:
                    print('skipped: empty output')
                    continue

                # Save: execution output
                output_path = os.path.join(checkpoint, 'output')
                os.makedirs(output_path, exist_ok=True)
                for file_name, output in outputs.items():
                    file_path = os.path.join(output_path, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(output)

                # Record: pass 2
                record[agent_name][f'task_{task_id}']['num_passes_2'] += 1

    summary = {
        'Agent Name': [],
        '(Run Level) Pass Rate 1': [],
        '(Run Level) Pass Rate 2': [],
        '(Task Level) Pass Rate 1': [],
        '(Task Level) Pass Rate 2': []
    }
    for agent_name, agent_record in record.items():
        num_runs, num_tasks = 0, len(agent_record)
        run_passes_1, run_passes_2 = 0, 0
        task_passes_1, task_passes_2 = 0, 0
        for task_record in agent_record.values():
            num_runs += task_record['num_runs']
            run_passes_1 += task_record['num_passes_1']
            run_passes_2 += task_record['num_passes_2']
            if task_record['num_passes_1'] > 0:
                task_passes_1 += 1
            if task_record['num_passes_2'] > 0:
                task_passes_2 += 1
        summary['Agent Name'].append(agent_name)
        summary['(Run Level) Pass Rate 1'].append(run_passes_1 / num_runs)
        summary['(Run Level) Pass Rate 2'].append(run_passes_2 / num_runs)
        summary['(Task Level) Pass Rate 1'].append(task_passes_1 / num_tasks)
        summary['(Task Level) Pass Rate 2'].append(task_passes_2 / num_tasks)
    summary = pd.DataFrame(summary)
    print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent_name',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--save_path',
        default='./checkpoint/benchmark',
        type=str
    )
    args = parser.parse_args()
    main(args)
