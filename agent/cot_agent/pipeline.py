import json
import logging
import hashlib

from utils.parser import parse_code_to_workflow
from utils.llm import invoke_completion

from agent.cot_agent.agent.generator import (
    get_generator_agent_prompt,
    parse_generator_agent_response
)


class CoTAgentPipeline:
    def __init__(
        self,
        save_path: str
    ):
        self.save_path = save_path

        logger_name = hashlib.md5(save_path.encode()).hexdigest()
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='[{asctime}] {levelname}: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{'
        )
        file_handler = logging.FileHandler(
            filename=f'{self.save_path}/run.log',
            mode='w'
        )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def __call__(self, query_text: str):
        # Start pipeline
        self.logger.info('Pipeline started')

        # Fetch requirement
        self.logger.info('Fetched requirement:')
        self.logger.info(f'  {query_text}')
        with open(f'{self.save_path}/query.md', 'w') as query_file:
            query_file.write(query_text)

        # Generate workflow
        self.logger.info('Generate workflow')
        message = get_generator_agent_prompt(
            query=query_text
        )
        self.logger.info(f'Input prompt:\n  {message}')
        answer, usage = invoke_completion(message)
        geneation = answer.content
        self.logger.info(f'Generated answer:\n  {geneation}')
        self.logger.info(f'Token usage:\n  {usage}')

        # Parse response
        plan, code = parse_generator_agent_response(geneation)
        self.logger.info(f'Parsed plan:\n  {plan}')
        self.logger.info(f'Parsed code:\n  {code}')

        # Refine workflow
        try:
            workflow = parse_code_to_workflow(code)
        except Exception as error:
            # maybe: raise RuntimeError('Failed to refine workflow') from error
            self.logger.error('Failed to generate workflow')
            return None

        # Save workflow
        self.logger.info(f'Parsed workflow:\n  {workflow}')
        with open(f'{self.save_path}/code.py', 'w') as code_file:
            code_file.write(code)
        with open(f'{self.save_path}/workflow.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file, indent=4)

        # Finish pipeline
        self.logger.info('Pipeline finished')
        return workflow
