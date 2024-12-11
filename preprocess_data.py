import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class PreprocessData:
    @classmethod
    def preprocess(data):
        """
        Data is a pandas dataframe with the dtypes
        """
        categorical_columns = []
        text_columns = []
        for column_data_type, column_name in zip(data.dtypes, data.columns):
            if type(column_data_type) == np.dtypes.ObjectDType:
                # Count unique columns to declare categorical
                num_unique = len(data[column_name].unique())
                if num_unique / len(data.index) < 0.1:
                    categorical_columns.append(column_name)
                else:
                    text_columns.append(column_name)
        # One hot encode the data
        one_hot_encoded = pd.get_dummies(data[categorical_columns], prefix=categorical_columns).astype(int)
        data = data.drop(columns=categorical_columns, axis=1).join(one_hot_encoded)
        # Tfidf and then SVD for the other columns
        for column in text_columns:
            transformer = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("best", TruncatedSVD(n_components=1)),
                ]
            )
            transformed_data = transformer.fit_transform(data[column])
            data[column] = transformed_data
        return data