import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network
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
    
# Define the function that trains the neural network
class NeuralNetworkWrapper:
    def __init__(self, input_size):
        # Constants that we define for algorithm
        learning_rate = 0.01
        hidden_size = 20
        self.epochs = 100
        # Define network
        self.model = SimpleFeedForwardNN(input_size, hidden_size=hidden_size, output_size=1)
        # Use MSE loss and stochastic gradient descent
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, inputs, targets):
        """
        Inputs would have input_size columns and however many rows
        Targets would have 1 column and however many rows
        """
        # Convert to tensors
        # Normalize inputs
        inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Can remove later when no need to print
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                return False
        return True

    def predict(self, inputs):
        """
        Inputs are the data we actually want to predict
        """
        # Make tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        self.model.eval()
        # Saves memory and computation time
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions