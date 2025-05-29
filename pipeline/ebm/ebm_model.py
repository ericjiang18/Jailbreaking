import torch
import torch.nn as nn

class SimpleEBM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Define a simple MLP for the energy function
        # Example: A few linear layers with non-linearities
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Or nn.ReLU(), nn.Tanh(), etc.
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) # Output a single scalar energy value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input activation vector(s) of shape (batch_size, input_dim)
                              or (input_dim) if a single vector.
        Returns:
            torch.Tensor: Scalar energy value(s) of shape (batch_size, 1) or (1).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0) # Add batch dimension if single vector
        energy = self.network(x)
        return energy


def load_ebm_model(model_path: str, input_dim: int, hidden_dim: int, device: str = 'cuda'):
    """
    Loads a trained EBM model from a checkpoint.
    """
    model = SimpleEBM(input_dim, hidden_dim)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"EBM model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading EBM model from {model_path}: {e}")
        return None
    return model
