"""
Optional Flower adapter. The simulation itself does NOT need flwr; this lets
the same HospitalClient run in a real Flower deployment (pip install flwr).
"""
import numpy as np
from module5_federated.fedavg import HospitalClient, local_train


def make_flower_client(client: HospitalClient, epochs=1, lr=0.05):
    import flwr as fl

    class _FlowerHospital(fl.client.NumPyClient):
        def get_parameters(self, config):
            d = client.X_train.shape[1]
            return [np.zeros(d), np.zeros(1)]

        def fit(self, parameters, config):
            w, b = parameters[0], float(parameters[1][0])
            w2, b2 = local_train(w, b, client.X_train, client.y_train, epochs=epochs, lr=lr)
            return [w2, np.array([b2])], client.n_samples, {}

        def evaluate(self, parameters, config):
            loss, acc = client.evaluate(parameters[0], float(parameters[1][0]))
            return loss, len(client.X_val), {'accuracy': acc}

    return _FlowerHospital()
