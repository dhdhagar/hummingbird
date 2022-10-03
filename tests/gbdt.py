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


class GBDTLayer(torch.nn.Module):
    """ Wrapper GBDT layer that trains a LightGBM model, converts initializes to torch using hummingbird """

    def __init__(self, X, y, n_estimators=10, random_state=1234, dropout=0.1):
        super().__init__()
        # Instantiate the LightGBM model
        lgbmclf = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
        # Train the LightGBM model to get the initial params
        lgbmclf.fit(X, y)
        # Convert the LightGBM model to PyTorch
        hbclf = hummingbird.ml.convert(lgbmclf, "torch", X, extra_config={constants.FINE_TUNE: True,
                                                                          constants.FINE_TUNE_DROPOUT_PROB: dropout})
        self = hbclf


class GBDTModel(torch.nn.Module):
    def __init__(self, X, y):
        super().__init__()
        self.gbdt = GBDTLayer(X, y)

    def forward(self, x):
        y_pred = self.gbdt(x)
        return y_pred


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
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=4, n_redundant=0, random_state=0, shuffle=False
        )

        # model = GradientBoostingClassifier(n_estimators=10, random_state=1234)
        # model = lgb.LGBMClassifier(n_estimators=10, random_state=1234)
        # model.fit(X, y)
        # torch_model = hummingbird.ml.convert(model, "torch", X, extra_config={constants.FINE_TUNE: True, constants.FINE_TUNE_DROPOUT_PROB: 0.1})
        torch_model = GBDTModel(X, y)
        assert torch_model is not None

        # Do fine tuning
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(torch_model.model.parameters(), lr=1e-3, weight_decay=5e-4)
        y_tensor = torch.from_numpy(y).float()

        print("Original loss: ", loss_fn(torch.from_numpy(model.predict_proba(X)[:, 1]).float(), y_tensor).item())
        with torch.no_grad():
            torch_model.model.eval()
            print("Fine-tuning starts from loss: ", loss_fn(torch_model.model(X)[1][:, 1], y_tensor).item())
        torch_model.model.train()

        for i in range(200):
            optimizer.zero_grad()
            y_ = torch_model.model(X)[1][:, 1]
            loss = loss_fn(y_, y_tensor)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                with torch.no_grad():
                    torch_model.model.eval()
                    print("Iteration ", i, ": ", loss_fn(torch_model.model(X)[1][:, 1], y_tensor).item())
                torch_model.model.train()

        with torch.no_grad():
            torch_model.model.eval()
            print("Fine-tuning done with loss: ", loss_fn(torch_model.model(X)[1][:, 1], y_tensor).item())


if __name__ == "__main__":
    testobj = TestSklearnGradientBoostingConverter()
    testobj.test_gbdt_classifier_fine_tune()

