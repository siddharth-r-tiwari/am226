# import openai
# import json
# import pandas as pd

# # Constants that we can alter later for LLM model
# LLM_MODEL = "gpt-3.5-turbo-1106"

# class GenerateSyntheticData:
#     def __init__(self, generation_instructions):
#         # Save an instance of generations of synthetic data with the system context
#         # NOTE: it must mention the json inputs and outputs to succeed
#         self.generation_instructions = generation_instructions

#     def to_message(self, data_sample):
#         """
#         Takes all the information and converts it straight into llm input
#         """
#         # TODO: verify if this actually works as an llm input (the json file)
#         prompt = data_sample.to_json()
#         messages = [
#             {"role": "system", "content": self.generation_instructions},
#             {"role": "user", "content": prompt},
#         ]
#         return messages
    
#     def validate_json(self, text):
#         try:
#             return json.loads(text)
#         except:
#             raise Exception("Json didn't generate correctly")
    
#     def predict(self, api_key, data_sample):
#         """
#         api_key: openai api key to be passed in from main runner file
#         data_sample: pandas df that we should base the generation data on
#         """
#         # Setup api key
#         openai.api_key = api_key
#         # Prepare response parameters
#         response_parameters = {
#             "model": LLM_MODEL,
#             "messages": self.to_message(data_sample),
#             "response_format": {"type": "json_object"}
#         }
#         # Get the response
#         response = openai.ChatCompletion.create(**response_parameters)
#         generated_json = response["choices"][0]["message"]["content"]
#         # Make sure the format is correct and raise error if not
#         final_json = self.validate_json(generated_json)
#         return pd.read_json(final_json)

import google.generativeai as genai
import json
import pandas as pd

LLM_MODEL = "gemini-1.5-flash"

class GenerateSyntheticData:
    def __init__(self, api_key):
        """
        Initialize the synthetic data generator with instructions and the model name.
        """
        # Configure the Gemini API key
        genai.configure(api_key=api_key)

    def to_prompt(self, rows, data_sample):
        """
        Generates a prompt for a generative AI model based on a DataFrame's schema.

        Args:
            df (pd.DataFrame): The input DataFrame to infer the schema.
            task_description (str): A description of the task you want the model to perform.

        Returns:
            str: A complete prompt with the inferred schema.
        """

        # Infer types from the DataFrame
        type_mapping = {
            "object": "str",
            "int64": "int",
            "float64": "float",
            "bool": "bool",
            "datetime64[ns]": "datetime",
            "category": "str",
        }
        
        # Build the TypedDict-like schema
        schema = {
            column: type_mapping.get(str(dtype), "str")
            for column, dtype in data_sample.dtypes.items()
        }

        # Generate the prompt
        schema_string = "\n".join(
            f"  '{col}': {type_}," for col, type_ in schema.items()
        )
        prompt = f"""Generate {rows} rows of synthetic data using this data frame. No code or explanations, just a JSON with values:

        Use this JSON schema:

        Rows = {{
        {schema_string}
        }}
        Return: list[Rows]
        """
        return prompt


    def predict(self, rows, data_sample):
        """
        Generate synthetic data using the Gemini API.
        """

        # Create the prompt
        prompt = self.to_prompt(rows, data_sample)

        # Generate content using the Gemini model
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt)

        # Extract and validate the generated JSON
        generated_json = response.text  # Assuming `response.text` contains the JSON

        # Convert to pandas DataFrame
        return generated_json
