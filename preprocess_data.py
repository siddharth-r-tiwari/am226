import numpy as np
import pandas as pd

class PreprocessData:
    def preprocess(self, data):
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
        one_hot_encoded = pd.get_dummies(data[categorical_columns], prefix=categorical_columns)
        data = data.drop(columns=categorical_columns, axis=1).join(one_hot_encoded)
        print(data.head())


        return

import pandas as pd

data = pd.read_csv("car.csv")

preprocessor = PreprocessData()
preprocessor.preprocess(data)

# Leave floats and ints