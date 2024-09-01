import os
import re
import ast
import json

from markdown_to_json import dictify


def extract_key_value_pair(text):
    key, value = text.split(':', 1)
    return key.strip(), value.strip()


def fetch_name_by_index(dictionary, index):
    for key, item in dictionary.items():
        if item['index'] == index:
            return key
    return None


def extract_nested_markdown_list(markdown):
    stack = [{}]
    depth = 0
    pattern = re.compile(r'( *)- ([^:\n]+)(?:: ([^\n]*))?\n?')
    for space, key, value in pattern.findall(markdown):
        indent = len(space)
        if indent > depth:
            assert not stack[-1]
        elif indent < depth:
            stack.pop()
        if value:
            stack[-1][key] = value
        else:
            stack[-1][key] = {}
            stack.append(stack[-1][key])
        depth = indent
    content = stack[0]
    return content


def extract_document_with_template(filename, content):
    header_pattern = re.compile(r'---\n.*?\n---', re.DOTALL)
    content = header_pattern.sub('', content).strip()
    lines = content.split('\n')
    node_name = filename
    node_description = lines[6]
    node_document = dictify('\n'.join(lines[7:]))
    parameters_list = []
    inputs_list = []
    outputs_list = []

    document = f'- `{node_name}`: {node_description}\n'
    parameter_types = ['INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COMBO[INT]', 'COMBO[FLOAT]', 'COMBO[BOOLEAN]', 'COMBO[STRING]']

    # inputs (required)
    if isinstance(node_document['Input types'], dict) and 'Required' in node_document['Input types']:
        inputs_document = node_document['Input types']['Required']

        if isinstance(inputs_document, list):
            for index in range(0, len(inputs_document), 2):
                input_name = inputs_document[index].strip('*')
                input_document = inputs_document[index + 1]
                if len(input_document) == 3:
                    input_description = input_document[0]
                    _, input_type = extract_key_value_pair(input_document[1])
                    if input_type.strip('`') in parameter_types:
                        parameters_list.append((input_name, input_description, input_type))
                    else:
                        inputs_list.append((input_name, input_description, input_type))

    # inputs (optional)
    if isinstance(node_document['Input types'], dict) and 'Optional' in node_document['Input types']:
        inputs_document = node_document['Input types']['Optional']

        if isinstance(inputs_document, list):
            for index in range(0, len(inputs_document), 2):
                input_name = inputs_document[index].strip('*')
                input_document = inputs_document[index + 1]
                if len(input_document) == 3:
                    input_description = input_document[0]
                    _, input_type = extract_key_value_pair(input_document[1])
                    if input_type.strip('`') in parameter_types:
                        parameters_list.append((input_name, input_description, input_type))
                    else:
                        inputs_list.append((input_name, input_description, input_type))

    # outputs
    outputs_document = node_document['Output types']
    if isinstance(outputs_document, list):
        for index in range(0, len(outputs_document), 2):
            output_name = outputs_document[index].strip('*')
            output_document = outputs_document[index + 1]
            if len(output_document) == 3:
                output_description = output_document[1]
                _, output_type = extract_key_value_pair(output_document[0])
                outputs_list.append((output_name, output_description, output_type))

    document += '    - Parameters:\n'
    for parameter_name, parameter_description, parameter_type in parameters_list:
        document += f'        - {parameter_name}: {parameter_description} Type should be {parameter_type}.\n'
    document += '    - Inputs:\n'
    for input_name, input_description, input_type in inputs_list:
        document += f'        - {input_name}: {input_description} Type should be {input_type}.\n'
    document += '    - Outputs:\n'
    for output_name, output_description, output_type in outputs_list:
        document += f'        - {output_name}: {output_description} Type should be {output_type}.\n'

    template = {'id': None, 'type': node_name, 'title': None, 'parameters': {}, 'inputs': {}, 'outputs': {}}
    for parameter_index, parameter_info in enumerate(parameters_list):
        parameter_name, _, _ = parameter_info
        template['parameters'][parameter_name.strip('`')] = {'index': parameter_index, 'value': None}
    for input_index, input_info in enumerate(inputs_list):
        input_name, _, input_type = input_info
        template['inputs'][input_name.strip('`')] = {'index': input_index, 'type': input_type.strip('`'), 'link': None}
    for output_index, output_info in enumerate(outputs_list):
        output_name, _, output_type = output_info
        template['outputs'][output_name.strip('`')] = {'index': output_index, 'type': output_type.strip('`'), 'links': []}

    return document, template


def parse_code_to_workflow(code):
    node_count = 0
    link_count = 0
    object_dict = {}
    tensor_dict = {}
    node_dict = {}
    link_dict = {}

    tree_root = ast.parse(code)

    for tree_node in tree_root.body:
        code_line = ast.unparse(tree_node).strip()
        function_name = None
        variable_list = []
        parameter_list = []

        if isinstance(tree_node, ast.Assign):
            assign_node = tree_node
        else:
            continue
        call_node = assign_node.value
        function_name = call_node.func.id

        target_list = assign_node.targets
        for target in target_list:
            if isinstance(target, ast.Name):
                variable_list.append(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    assert isinstance(element, ast.Name), f'code line {code_line}: unexpected target type {type(element)}'
                    variable_list.append(element.id)
            else:
                raise ValueError(f'code line {code_line}: unexpected target type {type(target)}')

        keyword_list = call_node.keywords
        for keyword in keyword_list:
            if isinstance(keyword.value, ast.Name):
                parameter_list.append((keyword.arg, keyword.value.id))
            elif isinstance(keyword.value, ast.Constant):
                parameter_list.append((keyword.arg, keyword.value.value))
            else:
                raise ValueError(f'code line {code_line}: unexpected keyword type {type(keyword.value)}')

        template_name = f'{function_name.replace(".", " ")}.json'

        if template_name in os.listdir('./dataset/docs/template/'):
            template_path = f'./dataset/docs/template/{template_name}'
            with open(template_path, 'r') as template_file:
                node = json.load(template_file)

            for parameter_name, parameter_value in parameter_list:
                assert parameter_name in node['parameters'], f'code line {code_line}: parameter {parameter_name} not found in node {function_name}'
                node['parameters'][parameter_name]['value'] = parameter_value

            node_count += 1
            node_id = node_count
            node['id'] = node_id
            node_name = variable_list[0]
            object_dict[node_name] = node_id
            node_dict[node_id] = node

        # invoke
        else:
            assert function_name in object_dict, f'code line {code_line}: function {function_name} not found'
            node_id = object_dict[function_name]
            node = node_dict[node_id]

            for parameter_name, parameter_value in parameter_list:
                assert parameter_name in node['inputs'], f'code line {code_line}: input {parameter_name} not found in node {function_name}'

                if parameter_value is None:
                    continue
                assert parameter_value in tensor_dict, f'code line {code_line}: variable {parameter_value} is used before defined'

                input_index = node['inputs'][parameter_name]['index']
                link_type = node['inputs'][parameter_name]['type']
                last_id, output_index = tensor_dict[parameter_value]
                last_node = node_dict[last_id]
                last_name = fetch_name_by_index(last_node['outputs'], output_index)
                last_type = last_node['outputs'][last_name]['type']
                assert link_type == last_type, f'code line {code_line}: type mismatch between {last_type} and {link_type}'

                link_count += 1
                link_id = link_count
                link = [last_id, output_index, node_id, input_index, link_type]
                link_dict[link_id] = link
                node['inputs'][parameter_name]['link'] = link_id
                last_node['outputs'][last_name]['links'].append(link_id)

            for output_index, variable in enumerate(variable_list):
                tensor_dict[variable] = (node_id, output_index)

    workflow = {
        'nodes': [],
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
        'version': '0.4'
    }

    for node_id, node in node_dict.items():
        node_info = {
            'id': node_id,
            'type': node['type'],
            'inputs': [],
            'outputs': [],
            'widgets_values': [],
        }

        for input_name, input in node['inputs'].items():
            node_info['inputs'].append({
                'name': input_name,
                'type': input['type'],
                'link': input['link'],
                'slot_index': input['index']
            })
        node_info['inputs'].sort(key=lambda x: x['slot_index'])

        for output_name, output in node['outputs'].items():
            node_info['outputs'].append({
                'name': output_name,
                'type': output['type'],
                'links': output['links'],
                'slot_index': output['index']
            })
        node_info['outputs'].sort(key=lambda x: x['slot_index'])

        parameter_list = list(node['parameters'].values())
        parameter_list.sort(key=lambda x: x['index'])
        for parameter in parameter_list:
            node_info['widgets_values'].append(parameter['value'])

        workflow['nodes'].append(node_info)

    for link_id, link in link_dict.items():
        workflow['links'].append([link_id] + link)

    return workflow


def parse_workflow_to_code(workflow):
    code = ''
    type_list = []
    node_dict = {}
    link_dict = {}

    code += '# create nodes by instantiation\n'
    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']
        node_name = f'{node_type.replace(" ", "_").lower()}_{node_id}'
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    parameter['value'] = node_info['widgets_values'][parameter['index']]
                    if isinstance(parameter['value'], str):
                        parameter['value'] = parameter['value'].replace('\n', ' ')
                        parameter['value'] = f'"""{parameter["value"]}"""'
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    if isinstance(parameter_value, str):
                        parameter_value = parameter_value.replace('\n', ' ')
                        parameter_value = f'"""{parameter_value}"""'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        parameter_list = []
        for key, value in node_template['parameters'].items():
            parameter_list.append(f'{key}={value["value"]}')
        code += f'{node_name} = {node_type.replace(" ", "_")}({", ".join(parameter_list)})\n'

        node_input = []
        if 'inputs' in node_info:
            node_input = [(item['name'], item['type'], item['link']) for item in node_info['inputs']]
        node_output = []
        if 'outputs' in node_info:
            node_output = [(item['name'], item['type'], item['links']) for item in node_info['outputs']]
        node_dict[node_id] = {'name': node_name, 'input': node_input, 'output': node_output}

    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        link_dict[link_id] = {'variable': None, 'source': source_id, 'target': target_id}

    code += '\n# link nodes by invocation\n'
    remain_node = list(node_dict.keys())
    while remain_node:
        for node_id in remain_node:
            flag = True
            node = node_dict[node_id]
            for _, _, link_id in node['input']:
                if link_id is None:
                    continue
                if link_dict[link_id]['variable'] is None:
                    flag = False
                    break

            if flag:
                remain_node.remove(node_id)

                parameter_list = []
                for input_name, _, input_link in node['input']:
                    if input_link is None:
                        input_value = 'None'
                    else:
                        input_value = link_dict[input_link]['variable']
                    parameter_list.append(f'{input_name}={input_value}')

                return_list = []
                for output_name, _, output_links in node['output']:
                    return_name = f'{output_name.replace(" ", "_").lower()}_{node_id}'
                    return_list.append(return_name)
                    if isinstance(output_links, list):
                        for link_id in output_links:
                            link_dict[link_id]['variable'] = return_name
                if not return_list:
                    return_list.append(f'result_{node_id}')

                code += f'{", ".join(return_list)} = {node["name"]}({", ".join(parameter_list)})\n'

    return code


def parse_markdown_to_workflow(markdown):
    type_list = []
    node_dict = {}
    link_dict = {}

    pattern = re.compile(r'- Nodes:\n(.*)- Links:\n(.*)', re.DOTALL)
    node_content, link_content = pattern.search(markdown).groups()
    node_content = extract_nested_markdown_list(node_content)
    link_content = extract_nested_markdown_list(link_content)

    for node_name, node_info in node_content.items():
        node_id = int(node_name[1:])
        node_type = node_info['node_type'][1:-1]
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node = json.load(template_file)
        node_output = {key.lower(): value for key, value in node['outputs'].items()}
        node['outputs'] = node_output

        node['id'] = node_id
        for key, value in node_info.items():
            if key == 'node_type':
                continue
            else:
                assert key in node['parameters'], f'parameter {key} not found in node {node_type}'
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                else:
                    value = eval(value)
                node['parameters'][key]['value'] = value

        node_dict[node_id] = node

    for link_name, link_info in link_content.items():
        link_id = int(link_name[1:])
        source_info, target_info = link_info.split(' -> ')
        source_name, source_port = source_info.split('.')
        target_name, target_port = target_info.split('.')

        source_id = int(source_name[1:])
        source_output = node_dict[source_id]['outputs'][source_port]['index']
        target_id = int(target_name[1:])
        target_input = node_dict[target_id]['inputs'][target_port]['index']
        link_type = node_dict[source_id]['outputs'][source_port]['type']

        node_dict[source_id]['outputs'][source_port]['links'].append(link_id)
        node_dict[target_id]['inputs'][target_port]['link'] = link_id
        link_dict[link_id] = [source_id, source_output, target_id, target_input, link_type]

    workflow = {
        'nodes': [],
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
        'version': '0.4'
    }

    for node_id, node in node_dict.items():
        node_info = {
            'id': node_id,
            'type': node['type'],
            'inputs': [],
            'outputs': [],
            'widgets_values': [],
        }

        for input_name, input in node['inputs'].items():
            node_info['inputs'].append({
                'name': input_name,
                'type': input['type'],
                'link': input['link'],
                'slot_index': input['index']
            })
        node_info['inputs'].sort(key=lambda x: x['slot_index'])

        for output_name, output in node['outputs'].items():
            node_info['outputs'].append({
                'name': output_name,
                'type': output['type'],
                'links': output['links'],
                'slot_index': output['index']
            })
        node_info['outputs'].sort(key=lambda x: x['slot_index'])

        parameter_list = list(node['parameters'].values())
        parameter_list.sort(key=lambda x: x['index'])
        for parameter in parameter_list:
            node_info['widgets_values'].append(parameter['value'])

        workflow['nodes'].append(node_info)

    for link_id, link in link_dict.items():
        workflow['links'].append([link_id] + link)

    return workflow


def parse_workflow_to_markdown(workflow):
    markdown = ''
    type_list = []
    node_dict = {}
    link_dict = {}

    markdown += '- Nodes:\n'
    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']
        node_name = f'N{node_id}'
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    parameter['value'] = node_info['widgets_values'][parameter['index']]
                    if isinstance(parameter['value'], str):
                        parameter['value'] = parameter['value'].replace('\n', ' ')
                        parameter['value'] = f'"{parameter["value"]}"'
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    if isinstance(parameter_value, str):
                        parameter_value = parameter_value.replace('\n', ' ')
                        parameter_value = f'"{parameter_value}"'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        markdown += f'    - {node_name}:\n        - node_type: "{node_type}"\n'
        for key, value in node_template['parameters'].items():
            markdown += f'        - {key}: {value["value"]}\n'

        node_input = []
        if 'inputs' in node_info:
            node_input = [(item['name'], item['type'], item['link']) for item in node_info['inputs']]
        node_output = []
        if 'outputs' in node_info:
            node_output = [(item['name'], item['type'], item['links']) for item in node_info['outputs']]
        node_dict[node_id] = {'name': node_name, 'input': node_input, 'output': node_output}

    markdown += '\n- Links:\n'
    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        link_dict[link_id] = {'variable': None, 'source': source_id, 'target': target_id}

        if source_id in node_dict and target_id in node_dict:
            link_name = f'L{link_id}'
            source_name = f'N{source_id}'
            target_name = f'N{target_id}'
            source_output = node_dict[source_id]['output'][source_output][0].lower()
            target_input = node_dict[target_id]['input'][target_input][0].lower()
            markdown += f'    - {link_name}: {source_name}.{source_output} -> {target_name}.{target_input}\n'

    return markdown


def parse_workflow_to_prompt(workflow):
    prompt = {}
    links = {}

    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        links[link_id] = {'source_id': source_id, 'source_output': source_output}

    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    parameter['value'] = node_info['widgets_values'][parameter['index']]
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        node_inputs = {}
        for key, value in node_template['parameters'].items():
            node_inputs[key] = value['value']
        if 'inputs' in node_info:
            for item in node_info['inputs']:
                if item['link'] is not None:
                    source_id = links[item['link']]['source_id']
                    source_output = links[item['link']]['source_output']
                    node_inputs[item['name']] = [str(source_id), source_output]

        prompt[str(node_id)] = {'inputs': node_inputs, 'class_type': node_type}

    return prompt
