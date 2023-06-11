import ast
import json
import logging
import os
import re

import nltk
import tiktoken
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def fix_mismatched_brackets(s):
    # Count the number of opening and closing brackets
    logging.info("Fixing mismatched brackets")
    opening_brackets = s.count("{")
    closing_brackets = s.count("}")
    # Add missing closing brackets
    if opening_brackets > closing_brackets:
        s += "}" * (opening_brackets - closing_brackets)
    # Remove extra closing brackets
    if closing_brackets > opening_brackets:
        s = re.sub(r"\}", "", s, count=(closing_brackets - opening_brackets))

    return s


def dirty_json_parser(dirty_json):
    logging.info("Parsing dirty JSON")
    try:
        fixed_json = fix_mismatched_brackets(dirty_json)
        parsed_json = ast.literal_eval(fixed_json)
        if isinstance(parsed_json, (list, dict)):
            logging.info("Dirty JSON parsed successfully...")
            return True, parsed_json
        else:
            # if ast.literal_eval does not return a list or dictionary, try json.loads
            parsed_json = json.loads(fixed_json)
            logging.info("Dirty JSON parsed successfully using json.loads...")
            return True, parsed_json
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dirty JSON: {e}")
        return False, e


def save_html_files(html_list, result_name):
    # Create the directory with the result name if it doesn't exist
    directory = os.path.join(os.getcwd(), result_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save each HTML file as slide_number.html in the created directory
    for index, html in enumerate(html_list):
        slide_number = index + 1
        file_name = f"slide_{slide_number}.html"
        file_path = os.path.join(directory, file_name)

        with open(file_path, "w") as file:
            file.write(html)


def _count_tokens(text: str) -> int:
    logging.info("Counting tokens")
    encoding = tiktoken.encoding_for_model("gpt-4")
    try:
        tokens = encoding.encode(text)
        logging.info(f"Token count: {len(tokens)}")
        return len(tokens)
    except Exception as e:
        print(f"Error: {e}")
        return 0


def save_json(json_data, result_name):
    logging.info(f"Saving JSON as {result_name}.json")
    with open(f"{result_name}.json", "w") as file:
        json.dump(json_data, file, indent=4)


def compress_text(text, max_tokens):
    tokens = word_tokenize(text)
    compressed_text = " ".join(tokens[:max_tokens])
    return compressed_text
