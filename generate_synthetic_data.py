import openai
import json
import pandas as pd

# Constants that we can alter later for LLM model
LLM_MODEL = "gpt-3.5-turbo-1106"

class GenerateSyntheticData:
    def __init__(self, generation_instructions):
        # Save an instance of generations of synthetic data with the system context
        # NOTE: it must mention the json inputs and outputs to succeed
        self.generation_instructions = generation_instructions

    def to_message(self, data_sample):
        """
        Takes all the information and converts it straight into llm input
        """
        # TODO: verify if this actually works as an llm input (the json file)
        prompt = data_sample.to_json()
        messages = [
            {"role": "system", "content": self.generation_instructions},
            {"role": "user", "content": prompt},
        ]
        return messages
    
    def validate_json(self, text):
        try:
            return json.loads(text)
        except:
            raise Exception("Json didn't generate correctly")
    
    def predict(self, api_key, data_sample):
        """
        api_key: openai api key to be passed in from main runner file
        data_sample: pandas df that we should base the generation data on
        """
        # Setup api key
        openai.api_key = api_key
        # Prepare response parameters
        response_parameters = {
            "model": LLM_MODEL,
            "messages": self.to_message(data_sample),
            "response_format": {"type": "json_object"}
        }
        # Get the response
        response = openai.ChatCompletion.create(**response_parameters)
        generated_json = response["choices"][0]["message"]["content"]
        # Make sure the format is correct and raise error if not
        final_json = self.validate_json(generated_json)
        return pd.read_json(final_json)
