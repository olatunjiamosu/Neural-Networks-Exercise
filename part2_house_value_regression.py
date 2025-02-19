import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Initialise preprocessing attributes
        self.label_binariser = preprocessing.LabelBinarizer()
        self.scalers = {}
        self.nb_epoch = nb_epoch

        # Preprocess input data to determine dimensions
        X, _ = self._preprocessor(x, training = True)

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),            
        )

        # Define loss function and optimiser
        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr = 0.001)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Convert to numpy arrays if needed
        if isinstance(x, pd.DataFrame):
            x = x.copy() # Avoid modifying original data

            # Handle missing values 
            x['total_bedrooms'].fillna(x['total_bedrooms'].mean(), inplace = True)

            # Handle categorical features
            if training:
                self.label_binariser.fit(x['ocean_proximity'])
            ocean_proximity_encoded = self.label_binariser.transform(x['ocean_proximity'])

            # Drop original categorical column
            x = x.drop('ocean_proximity', axis = 1)

            # Scale numerical features
            numerical_features = x.columns[x.dtypes != 'object']
            if training:
                self.scalers = {
                    column: preprocessing.MinMaxScaler()
                    for column in numerical_features
                }
            
            for column in numerical_features:
                if training:
                    x[column] = self.scalers[column].fit_transform(x[[column]])
                else:
                    x[column] = self.scalers[column].transform(x[[column]])  

            # Combine numerical and encoded categorical features
            X = np.hstack([x.values, ocean_proximity_encoded])

            # Convert to PyTorch tensor
            X = torch.FloatTensor(X)
    
        # Preprocess output data if provided
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.values
            if training:
                self.y_scaler = preprocessing.MinMaxScaler()
                y = self.y_scaler.fit_transform(y)
            else:
                y = self.y_scaler.transform(y)
            y = torch.FloatTensor(y)

        # Return preprocessed x and y, return None for y if it was None
        return X, (y if y is not None else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        # Training loop
        self.model.train()
        for epoch in range(self.nb_epoch):
            # Forward pass
            self.optimiser.zero_grad()
            outputs = self.model(X)

            # Compute loss
            loss = self.criterion(outputs, Y)

            # Backward pass and optimisation step 
            loss.backward()
            self.optimiser.step()

            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.nb_epoch}, Loss: {loss.item():.4f}")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()
