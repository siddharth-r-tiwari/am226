# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define the neural network
# class SimpleFeedForwardNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleFeedForwardNN, self).__init__()
#         self.hidden_layer = nn.Linear(input_size, hidden_size)
#         self.activation = nn.ReLU()
#         self.output_layer = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.hidden_layer(x)
#         x = self.activation(x)
#         x = self.output_layer(x)
#         return x
    
# # Define the function that trains the neural network
# class NeuralNetworkWrapper:
#     def __init__(self, input_size):
#         # Constants that we define for algorithm
#         learning_rate = 0.01
#         hidden_size = 20
#         self.epochs = 100
#         # Define network
#         self.model = SimpleFeedForwardNN(input_size, hidden_size=hidden_size, output_size=1)
#         # Use MSE loss and stochastic gradient descent
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

#     def train(self, inputs, targets):
#         """
#         Inputs would have input_size columns and however many rows
#         Targets would have 1 column and however many rows
#         """
#         for epoch in range(self.epochs):
#             # Forward pass
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, targets)
#             # Backwards pass
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # Can remove later when no need to print
#             if (epoch + 1) % 10 == 0:
#                 print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

#     def predict(self, inputs):
#         """
#         Inputs are the data we actually want to predict
#         """
#         self.model.eval()
#         # Saves memory and computation time
#         with torch.no_grad():
#             predictions = self.model(inputs)
#         return predictions

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedForwardNN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class NeuralNetworkWrapper:
    def __init__(self, input_size):
        # Constants for the neural network
        learning_rate = 0.01
        hidden_size = 20
        self.epochs = 100
        self.model = SimpleFeedForwardNN(input_size, hidden_size=hidden_size, output_size=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        # Store mappings for categorical values
        self.mappings = {}

    def _convert_to_numeric(self, data, column_names=None):
        """
        Converts all input data into numeric values suitable for PyTorch.
        - Handles one-dimensional and multi-dimensional data.
        - Non-numeric data is mapped to integers using reusable mappings.
        - Returns a numpy array of type float32.
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:  # Handle 1D array (e.g., targets)
                if not np.issubdtype(data.dtype, np.number):
                    # Create a mapping for targets if they are non-numeric
                    if "targets" not in self.mappings:
                        unique_values = set(data)
                        self.mappings["targets"] = {value: idx for idx, value in enumerate(unique_values)}
                    data = np.array([self.mappings["targets"][value] for value in data])
                return data.astype(np.float32)

            elif data.ndim == 2:  # Handle 2D array (e.g., inputs)
                processed_data = []
                for i, column in enumerate(data.T):
                    col_name = column_names[i] if column_names else f"col_{i}"
                    if not np.issubdtype(column.dtype, np.number):
                        # Create mapping for non-numeric data
                        if col_name not in self.mappings:
                            unique_values = set(column)
                            self.mappings[col_name] = {value: idx for idx, value in enumerate(unique_values)}
                        mapped_column = np.array([self.mappings[col_name][value] for value in column])
                        processed_data.append(mapped_column)
                    else:
                        processed_data.append(column.astype(float))
                return np.column_stack(processed_data).astype(np.float32)
            else:
                raise TypeError("Input data must be a numpy array.")

    def train(self, inputs, targets):
        """
        Inputs would have input_size columns and however many rows
        Targets would have 1 column and however many rows
        """
        # Convert inputs and targets to numerical format
        inputs = self._convert_to_numeric(inputs)
        targets = self._convert_to_numeric(targets).reshape(-1, 1)

        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Training loop
        for epoch in range(self.epochs):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def predict(self, inputs, column_names=None):
        """
        Inputs are the data we actually want to predict
        """
        # Convert inputs to numerical format
        inputs = self._convert_to_numeric(inputs, column_names)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions



# One hot encoding functions (untested)

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.decomposition import TruncatedSVD
# from sklearn.utils import check_random_state
# import pandas as pd
# import numpy as np

# class NeuralNetworkWrapper:
#     def __init__(self, input_size):
#         # Constants for the neural network
#         learning_rate = 0.01
#         hidden_size = 20
#         self.epochs = 100
#         self.model = SimpleFeedForwardNN(input_size, hidden_size=hidden_size, output_size=1)
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
#         self.mappings = {}

#     def _convert_to_numeric(self, data):
#         """
#         Converts input data into numerical format.
#         - Handles one-hot encoding for categorical data.
#         - Applies TFTDI for frequency-based encoding.
#         - Uses SVD for dimensionality reduction (if specified).
#         - Converts all numeric data to float32.
#         """
#         if isinstance(data, pd.DataFrame):
#             processed_data = []

#             # Separate categorical and numerical columns
#             categorical_cols = data.select_dtypes(include=['object', 'category']).columns
#             numerical_cols = data.select_dtypes(include=[np.number]).columns

#             # Process categorical columns
#             if len(categorical_cols) > 0:
#                 encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#                 one_hot_encoded = encoder.fit_transform(data[categorical_cols])

#                 # Apply SVD for dimensionality reduction (optional)
#                 if one_hot_encoded.shape[1] > 10:  # Example threshold for SVD
#                     svd = TruncatedSVD(n_components=min(10, one_hot_encoded.shape[1]))
#                     one_hot_encoded = svd.fit_transform(one_hot_encoded)

#                 processed_data.append(one_hot_encoded)

#             # Process numerical columns
#             if len(numerical_cols) > 0:
#                 numerical_data = data[numerical_cols].to_numpy().astype(np.float32)
#                 processed_data.append(numerical_data)

#             # Combine processed data
#             return np.hstack(processed_data).astype(np.float32)

#         elif isinstance(data, np.ndarray):
#             # If already a numpy array, ensure it's float32
#             return data.astype(np.float32)

#         elif isinstance(data, pd.Series):
#             # For single-column data, convert to numpy array and encode if needed
#             if data.dtype in ['object', 'category']:
#                 # Encode categorical data
#                 unique_values = data.unique()
#                 if data.name not in self.mappings:
#                     self.mappings[data.name] = {v: i for i, v in enumerate(unique_values)}
#                 return np.array([self.mappings[data.name][v] for v in data], dtype=np.float32)
#             else:
#                 # Convert numerical data
#                 return data.to_numpy().astype(np.float32)

#         else:
#             raise ValueError("Unsupported data type. Provide a pandas DataFrame, Series, or numpy array.")


# def train(self, inputs, targets):
#     inputs = self._convert_to_numeric(inputs)
#     targets = self._convert_to_numeric(targets).reshape(-1, 1)

#     inputs = torch.tensor(inputs, dtype=torch.float32)
#     targets = torch.tensor(targets, dtype=torch.float32)

#     for epoch in range(self.epochs):
#         outputs = self.model(inputs)
#         loss = self.criterion(outputs, targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

