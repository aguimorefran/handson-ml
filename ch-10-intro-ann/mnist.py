from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Load data
digits = load_digits()
data = digits.data

# Train, test, val

# Split data into train and test sets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(digits.target, test_size=0.2, random_state=42)

# Split data into train and validation sets
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)

# Print shapes
print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
print('Validation shape:', X_val.shape)

target_size = len(np.unique(digits.target))
data_size = digits.data.shape[1]
print('Input vector size:', data_size)
print('Target size:', target_size)

# Print dimensions of one sample
print('Sample dimensions:', digits.data[0].shape)
print('Sample:', digits.data[0])


# Create a MLP model, using GridSearchCV to find the best hyperparameters
def build_model():
    model = MLPClassifier()
    parameters = {
        'hidden_layer_sizes': [
            (300, 100,),
            (300, 300, 200,),
            (300, 300, 300, 200,),
        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'early_stopping': [True],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [1e-3, 1e-4, 1e-5],
    }

    clf = make_pipeline(StandardScaler(), GridSearchCV(model, parameters, cv=3, verbose=3))
    return clf


# Train the model
model = build_model()
print(model)

# Fit the model
model.fit(
    X=X_train,
    y=y_train,
)

# Print the best hyperparameters
print('Best hyperparameters:', model.named_steps['gridsearchcv'].best_params_)
print('Best score:', model.named_steps['gridsearchcv'].best_score_)
print('Validation score:', model.score(X_val, y_val))
