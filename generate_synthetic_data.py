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
    
    def get_schema(self, data_sample):
        # Map pandas dtypes to JSON Schema types
        type_mapping = {
            "object": "string",
            "int64": "integer",
            "float64": "number",
            "bool": "boolean",
            "datetime64[ns]": "string",  # Dates as ISO 8601 strings
            "category": "string",
        }
        
        properties = {
            column: {"type": type_mapping.get(str(dtype), "string")}
            for column, dtype in data_sample.dtypes.items()
        }
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys())
            }
        }
        return schema

    def to_prompt(self, rows, data_sample):
        """
        Generates a prompt for a generative AI model based on a DataFrame's schema.

        Args:
            df (pd.DataFrame): The input DataFrame to infer the schema.
            task_description (str): A description of the task you want the model to perform.

        Returns:
            str: A complete prompt with the inferred schema.
        """
        prompt = f"""Generate {rows} rows of synthetic data using this data frame. No code or explanations, just a JSON with values:"""
        return prompt


    def predict(self, row_multiplier, dataset_length, data_sample):
        """
        Generate synthetic data using the Gemini API.
        """
        rows = row_multiplier * dataset_length
        # Create the prompt
        prompt = self.to_prompt(rows, data_sample)
        # Generate content using the Gemini model
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=self.get_schema(data_sample)
            ),
        )

        # Extract and validate the generated JSON
        generated_json = response.text
        try: 
            data = json.loads(generated_json)
            return pd.DataFrame(data)
        except:
            print("NO VALID JSON (AH)")
            return pd.DataFrame()
    
    def validate(self, categorical_columns, original_dataset, synthetic_data):
        """
        Ensures that the cateogrical columns in the synthetic dataset are in the original 
        """
        # Find the categories that are wrong for the synthetic dataset
        bad_categories = {}
        for column in categorical_columns:
            allowed_categories = original_dataset[column].unique()
            synthetic_categories = synthetic_data[column].unique()
            for cat in synthetic_categories:
                if cat not in allowed_categories:
                    if column in bad_categories.keys():
                        bad_categories[column].append(cat)
                    else:
                        bad_categories[column] = [cat]
        # Remove those categories
        for column, invalid_cats in bad_categories.items():
            synthetic_data = synthetic_data[~synthetic_data[column].isin(invalid_cats)]
        return synthetic_data      
    
