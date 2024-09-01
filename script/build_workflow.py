import json

from utils.parser import parse_workflow_to_code, parse_workflow_to_markdown


def main():
    with open('./dataset/workflow/meta.json', 'r') as meta_file:
        metadata = json.load(meta_file)

    for name, path in metadata.items():
        print(f'building workflow: {name}')
        with open(path['workflow'], 'r') as workflow_file:
            workflow = json.load(workflow_file)

        code = parse_workflow_to_code(workflow)
        with open(path['code'], 'w') as code_file:
            code_file.write(code)

        markdown = parse_workflow_to_markdown(workflow)
        with open(path['markdown'], 'w') as markdown_file:
            markdown_file.write(markdown)


if __name__ == '__main__':
    main()
