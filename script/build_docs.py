import os
import glob
import json

from utils.parser import extract_document_with_template


def main():
    for filepath in glob.glob(f'./dataset/docs/salty/*/Nodes/*.md'):
        filename, _ = os.path.splitext(os.path.basename(filepath))
        codename = filename.replace(' ', '_')
        print(f'building docs: {filename}')

        with open(filepath, 'r', encoding='utf-8') as content_file:
            content = content_file.read()

        document, template = extract_document_with_template(filename, content)

        with open(f'./dataset/docs/node/{codename}.md', 'w', encoding='utf-8') as document_file:
            document_file.write(document)

        with open(f'./dataset/docs/template/{codename}.json', 'w', encoding='utf-8') as template_file:
            json.dump(template, template_file, indent=4)


if __name__ == '__main__':
    main()
