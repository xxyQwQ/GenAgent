import json
import logging
import hashlib

from utils.parser import parse_code_to_workflow
from utils.llm import retrieve_references, invoke_completion

from agent.single_gen_agent.utils.state import AgentState
from agent.single_gen_agent.agent.analyzer import get_analyzer_agent_prompt
from agent.single_gen_agent.agent.generator import (
    get_generator_agent_prompt,
    parse_generator_agent_response
)
from agent.single_gen_agent.agent.refiner import (
    get_refiner_agent_prompt,
    parse_refiner_agent_response
)


class SingleGenAgentPipeline:
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

        # Analyze requirement
        self.logger.info('Analyze requirement')

        message = get_analyzer_agent_prompt(
            query=query_text
        )
        self.logger.info(f'Input prompt:\n  {message}')
        answer, usage = invoke_completion(message)
        analysis = answer.content
        self.logger.info(f'Generated answer:\n  {analysis}')
        self.logger.info(f'Token usage:\n  {usage}')

        # Retrieve reference
        self.logger.info('Retrieve reference')

        references = retrieve_references(
            requirement=analysis,
            count=self.num_refs
        )
        self.logger.info('Retrieved references:')
        for reference in references:
            self.logger.info(f'  {reference.metadata["name"]}: {reference.page_content}')

        # Generate workflow
        self.logger.info('Generate workflow')

        state = AgentState()
        for generation_step in range(1, self.num_steps + 1):
            # Generate workflow
            message = get_generator_agent_prompt(
                query=query_text,
                analysis=analysis,
                references=references,
                state=state
            )
            self.logger.info(f'Input prompt:\n  {message}')
            answer, usage = invoke_completion(message)
            generation = answer.content
            self.logger.info(f'Generated answer:\n  {generation}')
            self.logger.info(f'Token usage:\n  {usage}')

            # Parse response
            plan, action, code, desc = parse_generator_agent_response(
                response=generation
            )
            self.logger.info(f'Parsed plan:\n  {plan}')
            self.logger.info(f'Parsed action:\n  {action}')
            self.logger.info(f'Parsed code:\n  {code}')
            self.logger.info(f'Parsed description:\n  {desc}')

            # Refine workflow
            for refinement_step in range(1, self.num_fixes + 1):
                try:
                    workflow = parse_code_to_workflow(code)
                    if workflow is not None:
                        break

                except Exception as error:
                    if refinement_step == self.num_fixes:
                        # maybe: raise RuntimeError('Failed to refine workflow') from error
                        self.logger.error('Failed to refine workflow')
                        return None

                    message = get_refiner_agent_prompt(
                        query=query_text,
                        analysis=analysis,
                        references=references,
                        state=state,
                        plan=plan,
                        code=code,
                        desc=desc,
                        error=error
                    )
                    self.logger.info(f'Input prompt:\n  {message}')
                    answer, usage = invoke_completion(message)
                    refinement = answer.content
                    self.logger.info(f'Generated answer:\n  {refinement}')
                    self.logger.info(f'Token usage:\n  {usage}')

                    expl, code, desc = parse_refiner_agent_response(
                        response=refinement
                    )
                    self.logger.info(f'Parsed explanation:\n  {expl}')
                    self.logger.info(f'Parsed code:\n  {code}')
                    self.logger.info(f'Parsed description:\n  {desc}')

            # Update state
            state.update(
                plan=plan,
                action=action,
                code=code,
                desc=desc
            )

            # Save workflow
            self.logger.info(f'Parsed workflow:\n  {workflow}')
            with open(f'{self.save_path}/code.py', 'w') as code_file:
                code_file.write(code)
            with open(f'{self.save_path}/workflow.json', 'w') as workflow_file:
                json.dump(workflow, workflow_file, indent=4)

            # Submit workflow
            if action == 'Submit':
                break

            # Reach limit
            if generation_step == self.num_steps:
                # maybe: raise RuntimeError('Failed to generate workflow')
                self.logger.error('Failed to generate workflow')
                return None

        # Finish pipeline
        self.logger.info('Pipeline finished')
        return workflow
