import os
import time
import argparse

from utils.comfy import execute_workflow

from agent.zero_shot_agent.pipeline import ZeroShotAgentPipeline
from agent.few_shot_agent.pipeline import FewShotAgentPipeline
from agent.cot_agent.pipeline import CoTAgentPipeline
from agent.rag_agent.pipeline import RAGAgentPipeline
from agent.gen_agent.pipeline import GenAgentPipeline
from agent.json_gen_agent.pipeline import JsonGenAgentPipeline
from agent.list_gen_agent.pipeline import ListGenAgentPipeline
from agent.code_gen_agent.pipeline import CodeGenAgentPipeline
from agent.single_gen_agent.pipeline import SingleGenAgentPipeline


def main(args):
    # Setup checkpoint
    if args.save_path is None:
        args.save_path = f'./checkpoint/{time.strftime('%Y-%m-%d-%H-%M-%S')}'
    os.makedirs(args.save_path, exist_ok=True)

    # Create pipeline
    if args.agent_name == 'zero_shot_agent':
        pipeline = ZeroShotAgentPipeline(
            save_path=args.save_path
        )
    elif args.agent_name == 'few_shot_agent':
        pipeline = FewShotAgentPipeline(
            save_path=args.save_path
        )
    elif args.agent_name == 'cot_agent':
        pipeline = CoTAgentPipeline(
            save_path=args.save_path
        )
    elif args.agent_name == 'rag_agent':
        pipeline = RAGAgentPipeline(
            save_path=args.save_path,
            num_refs=args.num_refs
        )
    elif args.agent_name == 'gen_agent':
        pipeline = GenAgentPipeline(
            save_path=args.save_path,
            num_steps=args.num_steps,
            num_refs=args.num_refs,
            num_fixes=args.num_fixes
        )
    elif args.agent_name == 'json_gen_agent':
        pipeline = JsonGenAgentPipeline(
            save_path=args.save_path,
            num_steps=args.num_steps,
            num_refs=args.num_refs,
            num_fixes=args.num_fixes
        )
    elif args.agent_name == 'list_gen_agent':
        pipeline = ListGenAgentPipeline(
            save_path=args.save_path,
            num_steps=args.num_steps,
            num_refs=args.num_refs,
            num_fixes=args.num_fixes
        )
    elif args.agent_name == 'code_gen_agent':
        pipeline = CodeGenAgentPipeline(
            save_path=args.save_path,
            num_steps=args.num_steps,
            num_refs=args.num_refs,
            num_fixes=args.num_fixes
        )
    elif args.agent_name == 'single_gen_agent':
        pipeline = SingleGenAgentPipeline(
            save_path=args.save_path,
            num_steps=args.num_steps,
            num_refs=args.num_refs,
            num_fixes=args.num_fixes
        )

    # Generate workflow
    workflow = pipeline(args.query_text)
    if workflow is None:
        print('failed to generate workflow')
        return

    # Execute workflow
    status, outputs = execute_workflow(workflow)
    print(f'execution status: {status}')
    os.makedirs(f'{args.save_path}/output', exist_ok=True)
    for filename, output in outputs.items():
        print(f'save file: {args.save_path}/output/{filename}')
        with open(f'{args.save_path}/output/{filename}', 'wb') as output_file:
            output_file.write(output)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--agent_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--num_refs', type=int, default=5)
    parser.add_argument('--num_fixes', type=int, default=1)
    args = parser.parse_args()
    main(args)
