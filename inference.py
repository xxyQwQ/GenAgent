import os
import yaml
import argparse

from agent.zero_shot_agent.pipeline import ZeroShotAgentPipeline
from agent.few_shot_agent.pipeline import FewShotAgentPipeline
from agent.cot_agent.pipeline import CoTAgentPipeline
from agent.rag_agent.pipeline import RAGAgentPipeline
from agent.gen_agent.pipeline import GenAgentPipeline
from agent.json_gen_agent.pipeline import JsonGenAgentPipeline
from agent.list_gen_agent.pipeline import ListGenAgentPipeline
from agent.code_gen_agent.pipeline import CodeGenAgentPipeline
from agent.single_gen_agent.pipeline import SingleGenAgentPipeline


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    proxy_config = config['proxy']

os.environ['http_proxy'] = proxy_config['http_proxy']
os.environ['https_proxy'] = proxy_config['https_proxy']


def main(args):
    with open('./dataset/query/meta.json', 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)

    for agent_name in args.agent_name:
        print(f'[Inference] agent {agent_name}')

        for task_id, task_info in metadata.items():
            query = task_info['content']
            print(f'[Inference] task {task_id}')

            for run_id in range(1, args.num_runs + 1):
                checkpoint = f'{args.save_path}/{agent_name}/task_{task_id}/run_{run_id:03d}'
                os.makedirs(checkpoint, exist_ok=True)
                print(f'[Inference] run {run_id}/{args.num_runs}')

                # Skip: already inferred
                log_path = os.path.join(checkpoint, 'run.log')
                if not args.force_run and os.path.exists(log_path):
                    print('skipped: already inferred')
                    continue

                # Create pipeline
                if agent_name == 'zero_shot_agent':
                    pipeline = ZeroShotAgentPipeline(
                        save_path=checkpoint
                    )
                elif agent_name == 'few_shot_agent':
                    pipeline = FewShotAgentPipeline(
                        save_path=checkpoint
                    )
                elif agent_name == 'cot_agent':
                    pipeline = CoTAgentPipeline(
                        save_path=checkpoint
                    )
                elif agent_name == 'rag_agent':
                    pipeline = RAGAgentPipeline(
                        save_path=checkpoint,
                        num_refs=args.num_refs
                    )
                elif agent_name == 'gen_agent':
                    pipeline = GenAgentPipeline(
                        save_path=checkpoint,
                        num_steps=args.num_steps,
                        num_refs=args.num_refs,
                        num_fixes=args.num_fixes
                    )
                elif agent_name == 'json_gen_agent':
                    pipeline = JsonGenAgentPipeline(
                        save_path=checkpoint,
                        num_steps=args.num_steps,
                        num_refs=args.num_refs,
                        num_fixes=args.num_fixes
                    )
                elif agent_name == 'list_gen_agent':
                    pipeline = ListGenAgentPipeline(
                        save_path=checkpoint,
                        num_steps=args.num_steps,
                        num_refs=args.num_refs,
                        num_fixes=args.num_fixes
                    )
                elif agent_name == 'code_gen_agent':
                    pipeline = CodeGenAgentPipeline(
                        save_path=checkpoint,
                        num_steps=args.num_steps,
                        num_refs=args.num_refs,
                        num_fixes=args.num_fixes
                    )
                elif agent_name == 'single_gen_agent':
                    pipeline = SingleGenAgentPipeline(
                        save_path=checkpoint,
                        num_steps=args.num_steps,
                        num_refs=args.num_refs,
                        num_fixes=args.num_fixes
                    )

                # Run pipeline
                try:
                    workflow = pipeline(query)
                except Exception as error:
                    workflow = None

                # Check: pipeline status
                if workflow is None:
                    print(f'done: pipeline failed')
                else:
                    print(f'done: pipeline succeeded')


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
    parser.add_argument(
        '--num_runs',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_steps',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_refs',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_fixes',
        default=1,
        type=int
    )
    parser.add_argument(
        '--force_run',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    main(args)
