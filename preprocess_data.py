import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class PreprocessData:
    def __init__(self, whole_dataset, target_column):
        self.categorical_columns = []
        self.text_columns = []
        self.target_column = target_column

        # Identify categorical and text columns
        for column_data_type, column_name in zip(whole_dataset.dtypes, whole_dataset.columns):
            if type(column_data_type) == np.dtypes.ObjectDType and column_name != self.target_column:
                num_unique = len(whole_dataset[column_name].unique())
                if num_unique / len(whole_dataset.index) < 0.1:
                    self.categorical_columns.append(column_name)
                else:
                    self.text_columns.append(column_name)

        # Create dummy columns for the categorical columns in the whole dataset
        self.dummy_columns = pd.get_dummies(
            whole_dataset[self.categorical_columns], drop_first=False
        ).columns

    def preprocess(self, data):
        # Generate one-hot encoded columns
        dummies = pd.get_dummies(data[self.categorical_columns], drop_first=False).astype(int)
        # Ensure all expected columns are present
        dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)

        # Add the one-hot encoded columns to the dataframe
        data = data.drop(columns=self.categorical_columns, axis=1).join(dummies)

        # Handle text columns with a placeholder if necessary
        for column in self.text_columns:
            transformer = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("best", TruncatedSVD(n_components=1)),
                ]
            )
            transformed_data = transformer.fit_transform(data[column])
            data[column] = transformed_data

        return data
