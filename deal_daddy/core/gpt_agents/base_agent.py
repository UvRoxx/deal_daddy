import json
import logging
from typing import Union

import openai
from gpt_structs import CODE_COMPLETION_RESPONSE
from rich.console import Console

from core.utils import _count_tokens, dirty_json_parser

console = Console()


class ChatGPTAgent:
    def __init__(self, api_key, engine="gpt-3.5-turbo") -> None:
        self.api_key = api_key
        self.engine = engine
        openai.api_key = api_key

    def _api_call(self, messages) -> str:
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
        )
        return self.get_response_from_openai_response(response)

    @staticmethod
    def get_message_from_openai_response(response) -> str:
        if response:
            choices = response.get("choices")
            if choices:
                choice = choices[0]
                message = choice.get("message")
                if message:
                    return message.get("content")

    def process_message(self, input_json) -> str:
        messages = json.loads(input_json)
        return json.dumps({"response": self.api_call(messages)})

    @staticmethod
    def _check_token_limit(messages) -> bool:
        logging.info("Checking token limit")
        total_tokens = sum(_count_tokens(message["content"]) for message in messages)
        if total_tokens > 4096:
            logging.warning("Token limit exceeded.")
            return False
        return True

    def api_call(self, messages, console_message="🤔 Thinking...") -> Union[str, None]:
        logging.info("Making API call to OpenAI")
        if not self._check_token_limit(messages):
            return None
        console = Console()
        with console.status(f"[bold green]{console_message} \n") as status:
            try:
                response = openai.ChatCompletion.create(
                    model=self.engine,
                    messages=messages,
                )
                status.stop()
                return self.get_response_from_openai_response(response)
            except Exception as e:
                logging.error(f"API call failed: {e}")
                return

    @staticmethod
    def get_response_from_openai_response(response):
        logging.info("Getting message from OpenAI response")
        try:
            if response:
                choices = response.get("choices")
                if choices:
                    choice = choices[0]
                    message = choice.get("message")
                    if message:
                        return message.get("content")
            elif type(response) == str:
                return response
        except Exception as e:
            logging.error(
                f"Error in getting message from OpenAI response: {response} with error: {e}",
            )
            return response

    def auto_fix_json(self, json_string, schema, error) -> Union[str, None]:
        system_message = (
            "This AI takes a JSON string and ensures that it"
            " is parseable and fully compliant with the provided schema. If an object"
            " or field specified in the schema isn't contained within the correct JSON,"
            " it is omitted. The function also escapes any double quotes within JSON"
            " string values to ensure that they are valid. If the JSON string contains"
            " any None or NaN values, they are replaced with null before being parsed."
        )
        prompt = f"""
                - This is the JSON that needs to be fixed: {json_string}
                - MUST BE PARSABLE IN PYTHON USING eg: parsed_json = ast.literal_eval(dirty_json)
                - AND RESPONSE MUST BE COMPLIANT WITH THIS SCHEMA:{schema}
                - WHEN TRYING TO PARSE THIS JSON, THE FOLLOWING ERROR OCCURRED: {error}
                - Please fix the JSON string so that it is parseable and fully compliant with the provided schema.
                - ONLY RETURN JSON RESPONSE. THIS RESPONSE MESSAGE WILL BE PASSED TO A JSON PARSER FUNCTION & IT MUST PASS
                - PLEASE RETURN FIXED JSON AND NOT ANY OTHER TEXT OR JSON FORMAT OR THE PROGRAM WILL FAIL
           """
        result_string = self.api_call(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "😅 Trying to fix JSON...",
        )
        try:
            passed, result_string = dirty_json_parser(result_string)
            if not passed:
                logging.error(f"JSON fix failed\n Result:{result_string}")
                return
            return result_string
        except json.JSONDecodeError:  # noqa: E722
            logging.error("JSON fix failed")
            return

    def ask_gpt(
        self,
        role_description: str,
        instruction: str,
        response_structure: dict,
        code: str,
    ):
        logging.info("Asking GPT")
        prompt = f"""
            - INSTRUCTIONS: {instruction} \n THIS IS THE CODE: {code}  \n -----END_CODE------- \n
            - PLEASE RETURN JSON RESPONSE. THIS RESPONSE MESSAGE WILL BE PASSED TO A JSON PARSER FUNCTION & IT MUST PASS
            - PLEASE RETURN FIXED JSON AND NOT ANY OTHER TEXT OR JSON FORMAT OR THE PROGRAM WILL FAIL
            - MUST FOLLOW THIS SCHEMA: {response_structure}
        """
        result_string = self.api_call(
            [
                {"role": "user", "content": role_description},
                {"role": "system", "content": prompt},
            ],
            "🤔 Thinking...",
        )
        if result_string:
            try:
                flag, v = dirty_json_parser(result_string)
                if flag is False:
                    fixed_json = self.auto_fix_json(
                        result_string,
                        CODE_COMPLETION_RESPONSE,
                        v,
                    )
                    return fixed_json
                logging.info("generation completed...")
                return v
            except Exception as e:
                logging.error(f"JSON failed to load: {e}")
                try:
                    logging.info("Trying to fix JSON")
                    fixed_json = self.auto_fix_json(
                        result_string,
                        CODE_COMPLETION_RESPONSE,
                        e,
                    )
                    return fixed_json
                except Exception as e:
                    logging.error(f" JSON failed to fix: {e}")
                    return
        else:
            logging.error(" generation failed...")
            return
