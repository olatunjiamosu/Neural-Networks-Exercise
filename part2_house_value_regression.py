import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, root_mean_squared_error 
import torch.utils.data
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, ParameterSampler, KFold
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor


class Regressor():
    def __init__(self, x, nb_epoch = 1000, learning_rate=0.001, hidden_sizes=(64,32), dropout_rate=0.2, optimizer = "adam", weight_decay=0, activation = "relu", scaler="minmax", batch_size=64):
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
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # This helps handle possible unseen categories in testing
        self.scalers = {}
        self.scaler_type = scaler
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        # Ensure x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input x must be a pandas DataFrame.")

        # Ensure hidden_sizes is a tuple or list
        if not isinstance(hidden_sizes, (tuple, list)):
            raise TypeError("hidden_sizes must be a tuple or list.")

        # Ensure nb_epoch is a positive integer
        if not isinstance(nb_epoch, int) or nb_epoch <= 0:
            raise ValueError("nb_epoch must be a positive integer.")

        # Ensure learning_rate is a positive float
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")

        # Ensure dropout_rate is between 0 and 1
        if not isinstance(dropout_rate, float) or dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1.")

        # Ensure batch_size is a positive integer
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        # Preprocess input data to determine dimensions
        X, _ = self._preprocessor(x, training = True)


        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Dynamic model architecture
        layers = []
        input_size = X.shape[1]
        for size in hidden_sizes:
            layers += [
                nn.Linear(input_size, size),
                nn.BatchNorm1d(size),
                self.activation,
                nn.Dropout(dropout_rate)
            ]
            input_size = size
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)

        # Optimiser
        if optimizer == "adam":
            self.optimiser = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer == "sgd":
            self.optimiser = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Loss function
        self.criterion = nn.MSELoss()

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
        
        #Ensuring x is a copy to avoid issues with the original dataset
        x=x.copy()

        # Scale numerical features
        numerical_features = x.columns[x.dtypes != 'object']
        if training:
            # Dynamically choose scaler type
            if self.scaler_type == "minmax":
                scaler_class = preprocessing.MinMaxScaler
            elif self.scaler_type == "standard":
                scaler_class = preprocessing.StandardScaler
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")

            # Initialise scalers for each numerical feature
            self.scalers = {
                column: scaler_class()
                for column in numerical_features
            }
            
            # Fit and transform training data
            for column in numerical_features:
                x.loc[:,column] = self.scalers[column].fit_transform(x[[column]])
        else:
            # Transform validation/test data
            for column in numerical_features:
                x.loc[:, column] = self.scalers[column].transform(x[[column]])  


   
        # Handle missing values 
        numerical_cols = x.select_dtypes(include=[np.number]).columns
        if training:
            self.na_medians = x [numerical_cols].median().to_dict() # Calculate median for each numerical column
        x[numerical_cols] =x[numerical_cols].fillna(self.na_medians) # Fill missing values with median

        # Handle categorical features
        if training:
            self.ohe.fit(x[["ocean_proximity"]])  # Fit on training data
        ocean_encoded = self.ohe.transform(x[["ocean_proximity"]])  # Handle unseen categories

        # Drop original categorical column
        x = x.drop('ocean_proximity', axis = 1)

    
        # Combine numerical and encoded categorical features
        X = np.hstack([x.values, ocean_encoded])

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
        
        # Split into train/validation sets 80/20
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

        # Convert to DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        #Initialising tracking variables
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        epochs_no_improve = 0 

        # Training loop
        self.model.train()
        for epoch in range(self.nb_epoch):
            epoch_train_loss = 0
            
            #Batch training
            for X_batch, Y_batch in train_loader:
                self.optimiser.zero_grad()
                outputs = self.model(X_batch)
                # Compute loss
                loss = self.criterion(outputs, Y_batch)

                # Backward pass and optimisation step 
                loss.backward()
                self.optimiser.step()
                epoch_train_loss += loss.item()
            
            #Calculate validation loss
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                val_pred = self.model(X_val) # Make predictions on validation set
                val_loss = self.criterion(val_pred, Y_val).item() # Compute loss
            
            #Track Losses
            avg_train_loss = epoch_train_loss/ len(train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            #Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Print both losses
            if (epoch + 1) % 100 == 0:
                print(
                   f"Epoch {epoch+1}/{self.nb_epoch} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f}"
                )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

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
        
        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = self.model(X)
            # Inverse transform the predictions
            predictions = self.y_scaler.inverse_transform(predictions.cpu().numpy()) # Convert to numpy array

        return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
Here is an example of how the Regressor class can be used to train a regressor model o
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = self.model(X)
        
        #convert to numpy arrays
        y_true = Y.numpy()
        y_pred = predictions.numpy()

        # Return evaluation metrics
        return { 
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "Explained Variance": explained_variance_score(y_true, y_pred)
        }
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



def perform_hyperparameter_search(x_train, y_train): 
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
    best_score = float("inf")
    best_params = {}
    all_raw_results = []
    
    # Expanded hyperparameter space
    param_distributions = {
        "learning_rate": [1e-2, 1e-3, 5e-4, 1e-4],  # 4x more options
        "batch_size": [32, 64, 128],                 # Test different batch sizes
        "nb_epoch": [100, 200],                      # Longer training possible
        "hidden_sizes": [
            (64, 32), (128, 64), 
            (256, 128), (64, 64, 32)                # Deeper architectures
        ],
        "dropout_rate": [0.1, 0.2, 0.3],            # Wider dropout range
        "optimizer": ["adam", "sgd"],                # Compare optimizers
        "weight_decay": [0, 1e-4, 1e-3],             # Stronger regularization
        "activation": ["relu", "sigmoid"],        # Test activation types
        "scaler": ["minmax", "standard"]             # Compare scaling methods
    }
    
     # Use randomised search 
    n_iter = 20  # Number of combinations to try
    search_space = ParameterSampler(
        param_distributions, 
        n_iter=n_iter,
        random_state=42
    )
    k=3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for params in search_space:
        try:
            fold_losses = []
            for train_idx, val_idx in kf.split(x_train):
                # Split data for this fold
                x_fold_train = x_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                x_fold_val = x_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                # Train model on fold
                regressor = Regressor(x_fold_train, **params)
                regressor.fit(x_fold_train, y_fold_train)
                
                # Get validation loss for this fold
                fold_loss = regressor.score(x_fold_val, y_fold_val)["MSE"]
                fold_losses.append(fold_loss)

            # Average validation loss across folds
            avg_val_loss = np.mean(fold_losses)
            
            # Update best parameters
            if avg_val_loss < best_score:
                best_score = avg_val_loss
                best_params = params.copy()
                print(f"New best: {best_params} | Val Loss: {best_score:.4f}")

            # Record results
            all_raw_results.append({
                **params,
                "status": "success",
                "val_loss": avg_val_loss
            })
            
        except Exception as e:
            all_raw_results.append({
                **params,
                "status": "failed", 
                "error": str(e),
                "val_loss":np.inf
            })

    if best_params:
        final_model = Regressor(x_train, **best_params)
        final_model.fit(x_train, y_train)
        save_regressor(final_model)
    else:       
        print("Warning: Hyperparameter search failed. No valid model saved.")

    return best_params, all_raw_results  # Return the chosen hyper parameters and raw results for graphing

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def prepare_visualization_data(raw_results):
    df = pd.DataFrame(raw_results)
    
    # Preserve original categorical columns for display
    df['scaler_str'] = df['scaler']
    df['activation_str'] = df['activation']
    df['optimizer_str'] = df['optimizer']
    
    # Encode categorical columns for plotting
    categorical_cols = ['scaler', 'activation', 'optimizer']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Handle variable-length hidden_sizes tuples
    layer_cols = []
    if 'hidden_sizes' in df.columns:
        # Get maximum layer depth in the results
        max_layers = max(len(tup) for tup in df['hidden_sizes'])
        
        # Create columns for each possible layer, padding with NaN if needed
        layer_cols = [f'hidden_size_{i+1}' for i in range(max_layers)]
        padded_data = [
            list(tup) + [np.nan] * (max_layers - len(tup)) 
            for tup in df['hidden_sizes']
        ]
        df[layer_cols] = pd.DataFrame(padded_data, index=df.index)
        df = df.drop('hidden_sizes', axis=1)
    
    # Ensure numeric types for all relevant columns
    base_numeric_cols = ['learning_rate', 'batch_size', 'nb_epoch',
                        'dropout_rate', 'weight_decay', 'val_loss']
    numeric_cols = base_numeric_cols + layer_cols  # Dynamic inclusion of layers
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean up non-numeric columns
    df = df.drop(
        ['status', 'error', 'scaler_str', 'activation_str', 'optimizer_str'],
        axis=1, 
        errors='ignore'
    )
    df = df[df["val_loss"] != np.inf]  # Remove placeholder failures
    df = df.dropna(subset=["val_loss"])  # Remove NaN
    
    return df

def plot_hyperparameter_heatmap(raw_results, param1, param2_prefix):
    df = prepare_visualization_data(raw_results)
    
    layer_cols = [col for col in df.columns if col.startswith(param2_prefix)]
    
    for p in layer_cols:
        plt.figure(figsize=(10, 6))
        pivot_table = df.pivot_table(values='val_loss', index=param1, columns=p, aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt=".4f")
        plt.title(f"Heatmap: {param1} vs {p}")
        plt.savefig(f"heatmap_{param1}_vs_{p}.png")
        plt.close()

def plot_training_curves(regressor):
    """Plot loss progression during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(regressor.train_losses, label='Training Loss')
    plt.plot(regressor.val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', bbox_inches='tight')

def plot_parallel_coordinates(raw_results):
    df = prepare_visualization_data(raw_results)
    
    # Extract layer columns dynamically
    layer_cols = [col for col in df.columns if col.startswith('hidden_size_')]
    
    plt.figure(figsize=(15, 10))
    pd.plotting.parallel_coordinates(
        df[layer_cols + ['val_loss']], 
        'val_loss', 
        colormap='viridis'
    )
    plt.xticks(rotation=45)
    plt.title("Hyperparameter Performance Trajectory")
    plt.savefig("parallel_coordinates.png")
def plot_hyperparameter_pairs(results_df):
    """Pairplot showing relationships between key parameters"""
    sns.pairplot(results_df[['learning_rate', 'batch_size', 'dropout_rate', 'val_loss']],
                diag_kind='kde',
                plot_kws={'alpha': 0.6},
                diag_kws={'fill': False})
    plt.suptitle('Pairwise Hyperparameter Relationships', y=1.02)
    plt.savefig('hyperparameter_pairs.png', bbox_inches='tight')

def plot_hyperparameter_importance(results_df):
    """Plot feature importance of hyperparameters"""
    X = results_df.drop('val_loss', axis=1).select_dtypes(include=np.number)
    y = results_df['val_loss']
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(10).plot(kind='barh')
    plt.title("Hyperparameter Importance Ranking")
    plt.xlabel("Importance Score")
    plt.savefig("hyperparameter_importance.png", bbox_inches='tight')
    plt.close()

def plot_layer_performance(results_df):
    """Visualize impact of hidden layer sizes dynamically"""
    layer_cols = [col for col in results_df.columns if col.startswith('hidden_size_')]
    
    plt.figure(figsize=(12, 4 * len(layer_cols)))
    
    for i, col in enumerate(layer_cols, 1):
        plt.subplot(len(layer_cols), 1, i)
        sns.boxplot(x=col, y='val_loss', data=results_df)
        plt.title(f"Impact of Hidden Layer {i} Size")
    
    plt.tight_layout()
    plt.savefig("layer_size_performance.png", bbox_inches='tight')
    plt.close()

def plot_enhanced_residuals(regressor, x_train, predictions, y_true):
    """Enhanced residual analysis with categorical feature"""
    try:
        residuals = y_true - predictions.ravel()
        ocean_prox = x_train['ocean_proximity'].values
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=predictions.ravel(), y=residuals, hue=ocean_prox, 
                        palette="viridis", alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.title("Residual Analysis by Ocean Proximity")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.legend(title='Ocean Proximity')
        plt.savefig("residuals_ocean_proximity.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not plot enhanced residuals: {str(e)}")

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




def example_main2():
    data = pd.read_csv("housing.csv")
    x_train = data.loc[:, data.columns != "median_house_value"]
    y_train = data.loc[:, ["median_house_value"]]
    
    # Perform hyperparameter search
    best_params, all_raw_results = perform_hyperparameter_search(x_train, y_train)
    print(f"Best Hyperparameters: {best_params}")
    
    # Generate analysis visualisations
    results_df = prepare_visualization_data(all_raw_results)
    
    
    plot_hyperparameter_importance(results_df)
    plot_layer_performance(results_df)
    
    
    
    plot_hyperparameter_heatmap(results_df, 'learning_rate', 'hidden_sizes')
    plot_parallel_coordinates(results_df)
    plot_hyperparameter_pairs(results_df)
    
    # Load best model
    regressor = load_regressor()
    plot_training_curves(regressor)
    
    # Model performance
    predictions = regressor.predict(x_train)
    y_true = y_train.values.ravel()
    
    # Enhanced prediction plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, predictions, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    r2 = r2_score(y_true, predictions)
    plt.title(f"Actual vs Predicted Prices (R² = {r2:.2f})")
    plt.savefig("actual_vs_predicted.png", bbox_inches='tight')
    plt.close()
    
    # Enhanced residual analysis
    plot_enhanced_residuals(regressor, x_train, predictions, y_true)
    
    # Final metrics
    error = regressor.score(x_train, y_train)
    print(f"\nFinal Model Metrics: {error}")


if __name__ == "__main__":
    example_main2()
