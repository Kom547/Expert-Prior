import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys

# Import Actor from existing script
sys.path.append(os.getcwd())
try:
    from expert_imitation_learning_MoE import Actor
except ImportError:
    print("Warning: Could not import Actor directly. Ensure you run this from the project root.")

def load_data(input_path):
    if input_path.endswith('.npz'):
        data = np.load(input_path, allow_pickle=True)
        # assuming 'obs' or something similar
        if 'obs' in data:
            return data['obs']
        else:
            return data[data.files[0]]
    else:
        raise ValueError("Only .npz files are currently supported")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_model_path", type=str, required=True, help="Directory containing ensemble_*.pth")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to input .npz data")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_count", type=int, default=6)
    args = parser.parse_args()

    # Load states
    states = load_data(args.input_data_path)
    state_tensor = torch.FloatTensor(states).to(args.device)
    state_shape = (state_tensor.shape[1],)
    
    # We'll use the first model for router weights, or average across ensemble.
    # The requirement doesn't specify, so let's use the first available model.
    model_file = f"{args.expert_model_path}/ensemble_1.pth"
    if not os.path.exists(model_file):
        # Fallback to direct path if it's a direct file
        if os.path.isfile(args.expert_model_path):
            model_file = args.expert_model_path
        else:
            raise FileNotFoundError(f"Cannot find model file at {model_file}")

    # Action dim is assumed to be 2 for most MoE models but could be 1. 
    # Try to load state_dict and infer action_dim or assume 2.
    # Let's peek into the state_dict to find out output shape of mean_layer.
    state_dict = torch.load(model_file, map_location=args.device, weights_only=True)
    action_dim = state_dict['experts.0.mean_layer.weight'].shape[0]
    num_experts = state_dict['gating_network.4.weight'].shape[0]

    model = Actor(state_shape, action_dim, num_experts=num_experts).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # Pass through gating network manually to get weights
        gating_logits = model.gating_network(state_tensor)
        weights = F.softmax(gating_logits, dim=1)
        
        # Dominant expert
        dominant_expert = torch.argmax(weights, dim=1)
        
        # Entropy
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
        
        # Expert outputs
        experts_outputs = [expert(state_tensor) for expert in model.experts]
        means = torch.stack([m for m, _ in experts_outputs], dim=1)
        
    weights_np = weights.cpu().numpy()
    dominant_np = dominant_expert.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    means_np = means.cpu().numpy()

    records = []
    for i in range(len(states)):
        record = {
            "sample_id": i,
            "dominant_expert": dominant_np[i],
            "entropy": entropy_np[i]
        }
        for j in range(num_experts):
            record[f"expert_{j}_weight"] = weights_np[i, j]
            # Add mean outputs
            for k in range(action_dim):
                record[f"expert_{j}_action_{k}_mean"] = means_np[i, j, k]
                
        records.append(record)
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved MoE router raw data to {args.output_csv}")

if __name__ == "__main__":
    main()
