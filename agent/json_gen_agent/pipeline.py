import json
import logging
import hashlib

from utils.llm import retrieve_references, invoke_completion

from agent.json_gen_agent.utils.state import AgentState
from agent.json_gen_agent.utils.function import fetch_reference_by_name
from agent.json_gen_agent.agent.analyzer import get_analyzer_agent_prompt
from agent.json_gen_agent.agent.planner import (
    get_planner_agent_prompt,
    parse_planner_agent_response,
    parse_planner_agent_action
)
from agent.json_gen_agent.agent.combiner import (
    get_combiner_agent_prompt,
    parse_combiner_agent_response
)
from agent.json_gen_agent.agent.adapter import (
    get_adapter_agent_prompt,
    parse_adapter_agent_response
)
from agent.json_gen_agent.agent.refiner import (
    get_refiner_agent_prompt,
    parse_refiner_agent_response
)


class JsonGenAgentPipeline:
    def __init__(
        self,
        save_path: str,
        num_steps: int = 5,
        num_refs: int = 5,
        num_fixes: int = 1
    ):
        self.save_path = save_path
        self.num_steps = num_steps
        self.num_refs = num_refs
        self.num_fixes = num_fixes

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

        # Initialize agent
        self.logger.info('Initialize agent')

        analyzer_message = get_analyzer_agent_prompt(
            query=query_text
        )
        self.logger.info(f'Input prompt:\n  {analyzer_message}')
        answer, usage = invoke_completion(analyzer_message)
        analyzer_response = answer.content
        self.logger.info(f'Generated answer:\n  {analyzer_response}')
        self.logger.info(f'Token usage:\n  {usage}')

        references = retrieve_references(
            requirement=analyzer_response,
            count=self.num_refs
        )
        self.logger.info('Retrieved references:')
        for reference in references:
            self.logger.info(f'  {reference.metadata["name"]}: {reference.page_content}')

        workspace = {}
        with open('./dataset/workflow/meta.json') as meta_file:
            metadata = json.load(meta_file)
        path = metadata['text_to_image']
        with open(path['workflow'], 'r') as json_file:
            workspace['workflow'] = json_file.read()
        with open(path['description'], 'r') as desc_file:
            workspace['description'] = desc_file.read()

        state = AgentState(
            step=self.num_steps,
            query=query_text,
            analysis=analyzer_response,
            reference=references,
            workspace=workspace
        )

        # Generate workflow
        self.logger.info('Generate workflow')

        for step_epoch in range(1, self.num_steps + 1):
            # Choose action
            planner_message = get_planner_agent_prompt(
                state=state
            )
            self.logger.info(f'Input prompt:\n  {planner_message}')
            answer, usage = invoke_completion(planner_message)
            planner_response = answer.content
            self.logger.info(f'Generated answer:\n  {planner_response}')
            self.logger.info(f'Token usage:\n  {usage}')
            self.logger.info(f'  {usage}')

            thought, plan, action = parse_planner_agent_response(planner_response)
            command, arguments = parse_planner_agent_action(action)
            self.logger.info(f'Parsed thought:\n  {thought}')
            self.logger.info(f'Parsed plan:\n  {plan}')
            self.logger.info(f'Parsed action:\n  {action}')
            self.logger.info(f'Parsed command:\n  {command}')
            self.logger.info(f'Parsed arguments:\n  {arguments}')

            # Update history
            state.update_history(thought, plan, action)

            # Action `load`
            if command == 'load':
                reference = fetch_reference_by_name(references, arguments['name'])
                if reference is None:
                    raise RuntimeError('Invalid reference')
                with open(reference.metadata['workflow'], 'r') as json_file:
                    workflow = json_file.read()
                with open(reference.metadata['description'], 'r') as desc_file:
                    description = desc_file.read()

                state.update_workspace(workflow, description)

            # Action `combine`
            elif command == 'combine':
                reference = fetch_reference_by_name(references, arguments['name'])
                if reference is None:
                    raise RuntimeError('Invalid name of reference')

                combiner_message = get_combiner_agent_prompt(
                    state=state,
                    planning=plan,
                    reference=reference
                )
                self.logger.info(f'Input prompt:\n  {combiner_message}')
                answer, usage = invoke_completion(combiner_message)
                combiner_response = answer.content
                self.logger.info(f'Generated answer:\n  {combiner_response}')
                self.logger.info(f'Token usage:\n  {usage}')

                workflow, description = parse_combiner_agent_response(combiner_response)
                self.logger.info(f'Parsed workflow:\n  {workflow}')
                self.logger.info(f'Parsed description:\n  {description}')

                state.update_workspace(workflow, description)

            # Action `adapt`
            elif command == 'adapt':
                adaptation = arguments['prompt']

                adapter_message = get_adapter_agent_prompt(
                    state=state,
                    planning=plan,
                    adaptation=adaptation
                )
                self.logger.info(f'Input prompt:\n  {adapter_message}')
                answer, usage = invoke_completion(adapter_message)
                adapter_response = answer.content
                self.logger.info(f'Generated answer:\n  {adapter_response}')
                self.logger.info(f'Token usage:\n  {usage}')

                workflow, description = parse_adapter_agent_response(adapter_response)
                self.logger.info(f'Parsed workflow:\n  {workflow}')
                self.logger.info(f'Parsed description:\n  {description}')

                state.update_workspace(workflow, description)

            # Action `retrieve`
            elif command == 'retrieve':
                retrieval = arguments['prompt']
                references = retrieve_references(
                    requirement=retrieval,
                    count=self.num_refs
                )
                self.logger.info('Retrieved references:')
                for reference in references:
                    self.logger.info(f'  {reference.metadata["name"]}: {reference.page_content}')

                state.update_reference(references)

            # Action `finish`
            elif command == 'finish':
                break

            # Make refinement
            for fix_epoch in range(self.num_fixes + 1):
                try:
                    data = json.loads(workflow)
                    if 'nodes' in data and 'links' in data:
                        break

                except Exception as error:
                    if fix_epoch == self.num_fixes:
                        # Maybe: raise RuntimeError('Failed to refine workflow') from error
                        self.logger.info('Failed to refine workflow')
                        return None

                    refiner_message = get_refiner_agent_prompt(
                        state=state,
                        planning=plan,
                        refinement=str(error)
                    )
                    self.logger.info(f'Input prompt:\n  {refiner_message}')
                    answer, usage = invoke_completion(refiner_message)
                    refiner_response = answer.content
                    self.logger.info(f'Generated answer:\n  {refiner_response}')
                    self.logger.info(f'Token usage:\n  {usage}')

                    explanation, workflow, description = parse_refiner_agent_response(refiner_response)
                    self.logger.info(f'Parsed explanation:\n  {explanation}')
                    self.logger.info(f'Parsed workflow:\n  {workflow}')
                    self.logger.info(f'Parsed description:\n  {description}')

                    state.update_workspace(workflow, description)

            # Save workflow
            self.logger.info(f'Parsed workflow:\n  {workflow}')
            with open(f'{self.save_path}/workflow.json', 'w') as workflow_file:
                json.dump(data, workflow_file, indent=4)

            # Reach limit
            if step_epoch == self.num_steps:
                # Maybe: raise RuntimeError('Failed to generate workflow')
                self.logger.info('Failed to generate workflow')
                return None

        # Finish pipeline
        self.logger.info('Pipeline finished')
        return workflow
