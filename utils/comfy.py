import json
import yaml
import uuid

import websocket
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils.parser import parse_workflow_to_prompt


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    comfyui_config = config['comfyui']

SERVER_ADDRESS = comfyui_config['server_address']
CLIENT_ID = str(uuid.uuid4())


def queue_prompt(prompt):
    request = Request(
        url=f'http://{SERVER_ADDRESS}/prompt',
        data=json.dumps({
            'prompt': prompt,
            'client_id': CLIENT_ID
        }).encode('utf-8')
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def fetch_history(prompt_id):
    with urlopen(f'http://{SERVER_ADDRESS}/history/{prompt_id}') as response:
        return json.loads(response.read())


def fetch_output(filename, subfolder):
    parameter = urlencode({
        'filename': filename,
        'subfolder': subfolder,
        'type': 'output'
    })
    with urlopen(f'http://{SERVER_ADDRESS}/view?{parameter}') as response:
        return response.read()


def execute_prompt(prompt):
    outputs = {}

    socket = websocket.WebSocket()
    socket.connect(f'ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}')

    prompt_id = queue_prompt(prompt)['prompt_id']
    while True:
        data = socket.recv()
        if isinstance(data, str):
            message = json.loads(data)
            if message['type'] == 'executing':
                message = message['data']
                if message['node'] is None and message['prompt_id'] == prompt_id:
                    break

    history = fetch_history(prompt_id)[prompt_id]
    for node_output in history['outputs'].values():
        for type_output in node_output.values():
            for spec_output in type_output:
                if isinstance(spec_output, dict) and spec_output['type'] == 'output':
                    output = fetch_output(spec_output['filename'], spec_output['subfolder'])
                    outputs[spec_output['filename']] = output

    status = history['status']
    return status, outputs


def execute_workflow(workflow):
    prompt = parse_workflow_to_prompt(workflow)
    return execute_prompt(prompt)


def main():
    with open('./test.json', 'r') as file:
        workflow = json.load(file)
    status, outputs = execute_workflow(workflow)
    print(f'Status: {status}')
    for filename, output in outputs.items():
        with open(f'./{filename}', 'wb') as f:
            f.write(output)


if __name__ == '__main__':
    main()
