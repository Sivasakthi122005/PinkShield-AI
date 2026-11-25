import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module5_federated.simulate import run_federated_simulation

print("Module 5 - Federated Learning Simulation (FedAvg, non-IID hospitals)")
print("=" * 60)
run_federated_simulation(num_rounds=15)
print("\nModule 5 Complete!")
