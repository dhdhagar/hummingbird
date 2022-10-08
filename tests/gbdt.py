"""
Classification task fine-tuning of a LightGBM hummingbird model and a vanilla MLP
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import math
import hummingbird.ml
from hummingbird.ml import constants
from tree_utils import gbdt_implementation_map
import lightgbm as lgb

from IPython import embed


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GBDTLayer(torch.nn.Module):
    """ Wrapper GBDT layer that trains a LightGBM model, converts initializes to torch using hummingbird """

    def __init__(self, X, y, n_estimators=10, random_state=1234, dropout=0.1):
        super().__init__()
        # Instantiate the LightGBM model
        lgbmclf = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=10, random_state=random_state)
        # Train the LightGBM model to get the initial params
        lgbmclf.fit(X, y)
        # Convert the LightGBM model to PyTorch
        hbclf = hummingbird.ml.convert(lgbmclf, "torch", X, extra_config={constants.FINE_TUNE: True,
                                                                          constants.FINE_TUNE_DROPOUT_PROB: dropout})
        self.model = hbclf.model

    def forward(self, x):
        return self.model(x)


class GBDTModel(torch.nn.Module):
    def __init__(self, X, y):
        super().__init__()
        self.model = GBDTLayer(X, y)
        # self.data_parallel = params.get("data_parallel")
        # if self.data_parallel:
        #     self.model = torch.nn.DataParallel(self.model)

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


class VanillaModel(torch.nn.Module):
    def __init__(self, n_estimators, n_parameters_per_estimator, n_features, max_tree_depth):
        super().__init__()
        out_layer1 = int(2**max_tree_depth - 1)
        out_layer2 = int(math.floor((n_parameters_per_estimator + n_features) / (2**max_tree_depth) - n_features))
        self.stack_per_estimator = torch.nn.ModuleList(torch.nn.Sequential(
            torch.nn.Linear(n_features, out_layer1),
            torch.nn.ReLU(),
            torch.nn.Linear(out_layer1, out_layer2),
            torch.nn.ReLU(),
            torch.nn.Linear(out_layer2, 1),
        ) for i in range(n_estimators))

    def forward(self, x):
        all_logits = [stack(x) for stack in self.stack_per_estimator]
        avg_logits = sum(all_logits)/len(self.stack_per_estimator)
        return torch.sigmoid(avg_logits)


class Experiments():
    def __init__(self):
        pass

    def create_dataset(self, n_samples=5000, n_features=10, n_informative=10, random_state=17, shuffle=True, test_size=0.3):
        # Create data for training
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
            random_state=random_state, shuffle=shuffle
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def create_gbdt_hummingbird_model(self):
        # Create a hummingbird LightGBM model
        model = GBDTModel(self.X_train, self.y_train)
        assert model is not None
        print(f"Created LightGBM hummingbird model")
        print(model)
        print(f"Parameter count: {count_parameters(model)}")
        return model

    def create_vanilla_model(self, n_parameters_per_estimator, n_estimators=10,
                             n_features=10, max_tree_depth=5):
        # Create an ensemble model of fully-connected 3-layer MLPs
        model = VanillaModel(n_estimators=n_estimators, n_parameters_per_estimator=n_parameters_per_estimator,
                             n_features=n_features, max_tree_depth=max_tree_depth)
        assert model is not None
        print(f"Created vanilla model")
        print(model)
        print(f"Parameter count: {count_parameters(model)}")
        return model

    def create_vanilla_model_from_model(self, model_to_mimic, n_estimators=10,
                                         n_features=10, max_tree_depth=5):
        # Create a vanilla model based on the parameter count of another model
        n_parameters = count_parameters(model_to_mimic)
        n_parameters_per_estimator = n_parameters // n_estimators
        model = self.create_vanilla_model(n_parameters_per_estimator, n_estimators=n_estimators,
                                          n_features=n_features, max_tree_depth=max_tree_depth)
        return model

    def fine_tune_model(self, model, loss_fn, lr, weight_decay, predict_fn, iterations):
        # Fine tune torch model
        y_tensor = torch.from_numpy(self.y_train).float()
        model_parameters = None
        try:
            model_parameters = model.parameters()
        except:
            try:
                model_parameters = model.model.parameters()
            except:
                raise ValueError("Error: Model parameters could not be found")
        
        optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        print("\nStarting fine-tuning:\n")
        with torch.no_grad():
            torch_model.eval()
            print("Initial loss = ", loss_fn(predict_fn(model, self.X_train), y_tensor).item())
        model.train()
        
        for i in tqdm(range(iterations)):
            # Full gradient descent
            optimizer.zero_grad()
            y_ = predict_fn(model, self.X_train)
            loss = loss_fn(y_, y_tensor)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                with torch.no_grad():
                    model.eval()
                    print("Iteration ", i, ": ", loss_fn(predict_fn(model, self.X_train), y_tensor).item())
                model.train()
        with torch.no_grad():
            model.eval()
            print("Fine-tuning done with loss = ", loss_fn(predict_fn(model, self.X_train), y_tensor).item())


    def fine_tune_gbdt(self, 
                       model,
                       loss_fn=torch.nn.BCELoss(), 
                       lr=1e-3, 
                       weight_decay=5e-4,
                       predict_fn=lambda model, inputs: torch.flatten(model(inputs)[1][:, 1]),
                       iterations=200):
        print("\nFINE-TUNING GBDT:\n-----------------\n")
        self.fine_tune_model(model,
                             loss_fn=loss_fn, 
                             lr=lr, 
                             weight_decay=weight_decay,
                             predict_fn=predict_fn,
                             iterations=iterations)

    def fine_tune_vanilla(self, 
                          model,
                          loss_fn=torch.nn.BCELoss(), 
                          lr=1e-3, 
                          weight_decay=5e-4,
                          predict_fn=lambda model, inputs: torch.flatten(model(inputs)),
                          iterations=200):
        print("\nFINE-TUNING VANILLA:\n-----------------\n")
        self.fine_tune_model(model,
                             loss_fn=loss_fn, 
                             lr=lr, 
                             weight_decay=weight_decay,
                             predict_fn=predict_fn,
                             iterations=iterations)


if __name__ == "__main__":
    experiments = Experiments()
    experiments.create_dataset()
    
    # LightGBM Hummingbird
    gbdt = experiments.create_gbdt_hummingbird_model()
    experiments.fine_tune_gbdt(gbdt)

    # Vanilla MLP ensemble created to mimic the LGBM parameter count
    mlp_ensemble = experiments.create_vanilla_model_from_model(gbdt)
    experiments.fine_tune_vanilla(mlp_ensemble)
