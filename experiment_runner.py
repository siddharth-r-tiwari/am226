from sklearn.metrics import mean_squared_error
import torch
from neural_network import NeuralNetworkWrapper
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis

class ExperimentRunner:
    def __init__(self, dataset_schema, target_column):
        """
        This function creates the neural network that we will be benchmarking
        - It will be a single layer perceptron for simplicity
        - dataset_schema will be used to get the dimensions for the neural network
        - target column will tell us how to get input and target data
        """
        # Convert the data to a numpy array
        data_array = dataset_schema.to_numpy()
        # Input size is the number of columns in the dataset without the target column
        input_size = data_array.shape[1] - 1
        # Initialize a neural network and save target column
        self.neural_network = NeuralNetworkWrapper(input_size)
        self.target_column = target_column
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')


    def compute_characteristics(self, full_data):
        """
        This takes in a dataset and gets statistics on it for us to classify.
        Automatically determines if the column is numerical or string.
        """
        characteristics = []

        # Iterate through each column in the dataset
        for column in full_data.columns:
            col_data = full_data[column].dropna()
            if col_data.dtype in [np.number, 'int64', 'float64']:  # Numerical data
                stats = {
                    'column': column,
                    'source_type': 'number',
                    'mean': col_data.mean(),
                    'variance': col_data.var(),
                    'skew': skew(col_data),
                    'kurtosis': kurtosis(col_data),
                    'range': col_data.max() - col_data.min()
                }
            elif col_data.dtype in [object, 'string']:  # String data
                strings = col_data.unique()
                if len(strings) > 1:
                    # Compute embeddings
                    embeddings = self.embeddings.encode(strings)

                    # Compute pairwise distances
                    pairwise_distances = pdist(embeddings, metric='cosine')
                    similarity_scores = 1 - pairwise_distances

                    stats = {
                        'column': column,
                        'source_type': 'object',
                        'mean': np.mean(similarity_scores),
                        'variance': np.var(similarity_scores),
                        'skew': skew(similarity_scores),
                        'kurtosis': kurtosis(similarity_scores),
                        'range': np.max(similarity_scores) - np.min(similarity_scores)
                    }
                else:
                    stats = {
                        'column': column,
                        'source_type': 'object',
                        'mean': None,
                        'variance': None,
                        'skew': None,
                        'kurtosis': None,
                        'range': None
                    }
            else:
                stats = {
                    'column': column,
                    'source_type': 'unknown',
                    'mean': None,
                    'variance': None,
                    'skew': None,
                    'kurtosis': None,
                    'range': None
                }
            characteristics.append(stats)

        # Convert list of stats to DataFrame
        return pd.DataFrame(characteristics)



    def train_network(self, train_data):
        """
        This uses the data to train the network
        - Sometimes train_data will include generated information
        """
        # Use train data to train the neural network
        inputs = train_data.drop(columns=[self.target_column]).to_numpy()
        targets = train_data[self.target_column].to_numpy()

        # Pass this information into the neural network
        self.neural_network.train(inputs, targets)

    def benchmark_network(self, test_data):
        """
        Take in test data and calculate the score
        - test_data will never contain generated information
        """
        # Split test data
        inputs = test_data.drop(columns=[self.target_column]).to_numpy()
        targets = test_data[self.target_column].to_numpy()

        # Get the predictions
        predictions = self.neural_network.predict(inputs).detach().numpy()

        # Benchmark using the mean squared error
        mse = mean_squared_error(targets, predictions)
        return mse

    
