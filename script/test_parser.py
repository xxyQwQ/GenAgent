import os
import json
import shutil
import argparse

from utils.parser import (
    parse_workflow_to_code,
    parse_code_to_workflow,
    parse_workflow_to_markdown,
    parse_markdown_to_workflow,
    parse_workflow_to_prompt
)
from utils.comfy import execute_prompt


def main():
    parser = argparse.ArgumentParser(description='Test parser')
    parser.add_argument(
        '--force_retest',
        action='store_true',
        default=False,
        help='Force to test all the workflows'
    )
    args = parser.parse_args()

    with open('./dataset/workflow/meta.json', 'r') as meta_file:
        metadata = json.load(meta_file)

    for name, path in metadata.items():
        print(f'[{name}] test started')
        prefix = f'./script/test_parser/{name}'

        if os.path.exists(prefix):
            if args.force_retest:
                print(f'[{name}] cleaning up')
                shutil.rmtree(prefix)
            else:
                print(f'[{name}] already tested')
                continue
        os.makedirs(prefix, exist_ok=True)

        print(f'[test 1] code representation')
        with open(path['workflow'], 'r') as workflow_file:
            workflow = json.load(workflow_file)

        print(f'[step 1] workflow -> code')
        try:
            code = parse_workflow_to_code(workflow)
        except Exception as error:
            print(f'[step 1] failed: {error}')
            continue
        print(f'[step 1] succeeded')

        print(f'[step 2] code -> workflow')
        try:
            workflow = parse_code_to_workflow(code)
        except Exception as error:
            print(f'[step 2] failed: {error}')
            continue
        print(f'[step 2] succeeded')

        print(f'[step 3] workflow -> prompt')
        try:
            prompt = parse_workflow_to_prompt(workflow)
        except Exception as error:
            print(f'[step 3] failed: {error}')
            continue
        print(f'[step 3] succeeded')

        print(f'[step 4] prompt -> execution')
        try:
            status, outputs = execute_prompt(prompt)
            if status['status_str'] != 'success':
                raise Exception(f'{status}')
            for filename, output in outputs.items():
                with open(f'{prefix}/{filename}', 'wb') as output_file:
                    output_file.write(output)
        except Exception as error:
            print(f'[step 4] failed: {error}')
            continue
        print(f'[step 4] succeeded')

        print(f'[test 2] markdown representation')
        with open(path['workflow'], 'r') as workflow_file:
            workflow = json.load(workflow_file)

        print(f'[step 1] workflow -> markdown')
        try:
            markdown = parse_workflow_to_markdown(workflow)
        except Exception as error:
            print(f'[step 1] failed: {error}')
            continue
        print(f'[step 1] succeeded')

        print(f'[step 2] markdown -> workflow')
        try:
            workflow = parse_markdown_to_workflow(markdown)
        except Exception as error:
            print(f'[step 2] failed: {error}')
            continue
        print(f'[step 2] succeeded')

        print(f'[step 3] workflow -> prompt')
        try:
            prompt = parse_workflow_to_prompt(workflow)
        except Exception as error:
            print(f'[step 3] failed: {error}')
            continue
        print(f'[step 3] succeeded')

        print(f'[step 4] prompt -> execution')
        try:
            status, outputs = execute_prompt(prompt)
            if status['status_str'] != 'success':
                raise Exception(f'{status}')
            for filename, output in outputs.items():
                with open(f'{prefix}/{filename}', 'wb') as output_file:
                    output_file.write(output)
        except Exception as error:
            print(f'[step 4] failed: {error}')
            continue
        print(f'[step 4] succeeded')

        print(f'[{name}] test passed')


if __name__ == '__main__':
    main()
