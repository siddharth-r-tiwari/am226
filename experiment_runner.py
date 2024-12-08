from sklearn.metrics import mean_squared_error
from neural_network import NeuralNetworkWrapper

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

    def get_dataset_sample_information(self, full_data):
        """
        This takes in a dataset and gets statastics on it for us to classify
        """
        # TODO: decide which metrics to actually do this for
        # TODO: Sidd, let me know what you had in mind!
        pass

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
        predictions = self.neural_network.predict(inputs)
        # Benchmark using the mean squared error
        mse = mean_squared_error(targets, predictions)
        return mse

    