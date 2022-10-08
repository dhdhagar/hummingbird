"""
Tests Sklearn RandomForest, DecisionTree, ExtraTrees converters.
"""
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import torch

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


class LGBMWrapperModel(torch.nn.Module):
    def __init__(self, X, y, params):
        super().__init__()
        self.model = GBDTLayer(X, y)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


class ScratchModel(nn.Module):
    def __init__(self, n_estimators, n_parameters, n_features, tree_depth):
        super().__init__()
        out_layer1 = 2**tree_depth - 1
        out_layer2 = (n_parameters + n_features) / (2**tree_depth) - n_features
        self.stack_per_estimator = [nn.Sequential(
            nn.Linear(n_features, out_layer1),
            nn.ReLU(),
            nn.Linear(out_layer1, out_layer2),
            nn.ReLU(),
            nn.Linear(out_layer2, 1),
        ) for i in range(n_estimators)]

    def forward(self, x):
        all_logits = [stack(x) for stack in self.stack_per_estimator]
        avg_logits = sum(all_logits)/len(self.stack_per_estimator)
        return avg_logits


class TestSklearnGradientBoostingConverter():
    # Check tree implementation
    def test_gbdt_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)

        model = GradientBoostingClassifier(n_estimators=1, max_depth=1)
        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch", extra_config={constants.FINE_TUNE: True})
        self.assertIsNotNone(torch_model)
        self.assertTrue(str(type(list(torch_model.model._operators)[0])) == gbdt_implementation_map["gemm_fine_tune"])

    # Fine tune GBDT binary classifier.
    def test_gbdt_classifier_fine_tune(self):
        # Create data for training
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=4, n_redundant=0, random_state=0, shuffle=False
        )

        # Create a hummingbird LGBM model
        torch_model = LGBMWrapperModel(X, y)
        assert torch_model is not None
        print(f"Created LGBM hummingbird model")
        print(torch_model)
        print(f"Parameter count: {count_parameters(torch_model)}")

        # Create a from-scratch model to mimic the hummingbird parameter count and structure
        n_parameters = count_parameters(torch_model)
        n_estimators = 10
        n_parameters_per_estimator = n_parameters // n_estimators
        n_features = 10
        max_tree_depth = 10
        torch_model_scratch = ScratchModel(n_estimators=n_estimators, n_parameters=n_parameters_per_estimator,
                                           n_features=n_features, tree_depth=max_tree_depth)
        assert torch_model_scratch is not None
        print(f"Created scratch model")
        print(torch_model_scratch)
        print(f"Parameter count: {count_parameters(torch_model_scratch)}")

        # Do fine tuning on hummingbird
        print("\nFINE TUNING HUMMINGBIRD:\n----------------------\n")
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(torch_model.model.parameters(), lr=1e-3, weight_decay=5e-4)
        y_tensor = torch.from_numpy(y).float()
        with torch.no_grad():
            torch_model.eval()
            print("Fine-tuning starts from loss: ", loss_fn(torch_model(X)[1][:, 1], y_tensor).item())
        torch_model.train()
        for i in range(200):
            optimizer.zero_grad()
            y_ = torch_model(X)[1][:, 1]
            loss = loss_fn(y_, y_tensor)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                with torch.no_grad():
                    torch_model.eval()
                    print("Iteration ", i, ": ", loss_fn(torch_model(X)[1][:, 1], y_tensor).item())
                torch_model.train()
        with torch.no_grad():
            torch_model.eval()
            print("Fine-tuning done with loss: ", loss_fn(torch_model(X)[1][:, 1], y_tensor).item())

        # Do fine tuning on from-scratch model
        loss = None
        print("\nFINE TUNING FROM-SCRATCH MODEL:\n----------------------\n")
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(torch_model_scratch.parameters(), lr=1e-3, weight_decay=5e-4)
        y_tensor = torch.from_numpy(y).float()
        with torch.no_grad():
            torch_model_scratch.eval()
            print("Fine-tuning starts from loss: ", loss_fn(torch_model_scratch(X)[1][:, 1], y_tensor).item())
        torch_model_scratch.train()
        for i in range(200):
            optimizer.zero_grad()
            y_ = torch_model_scratch(X)[1][:, 1]
            loss = loss_fn(y_, y_tensor)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                with torch.no_grad():
                    torch_model_scratch.eval()
                    print("Iteration ", i, ": ", loss_fn(torch_model_scratch(X), y_tensor).item())
                torch_model_scratch.train()
        with torch.no_grad():
            torch_model_scratch.eval()
            print("Fine-tuning done with loss: ", loss_fn(torch_model_scratch(X), y_tensor).item())

if __name__ == "__main__":
    testobj = TestSklearnGradientBoostingConverter()
    testobj.test_gbdt_classifier_fine_tune()

