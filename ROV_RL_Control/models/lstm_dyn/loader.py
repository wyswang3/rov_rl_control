# models/lstm_dyn/loader.py
"""
Loader for the pretrained LSTM-based dynamics model.
Assumes the entire nn.Module was saved via torch.save(model, ...).
Provides load_dynamics(device) -> (model, init_hidden_fn).
"""
import torch
from pathlib import Path

# Path to the saved model file
MODEL_FILE = Path(__file__).resolve().parent / "model_hybrid_full.pt"

def load_dynamics(device: str = "cuda"):
    """
    Load the pretrained dynamics model and return it along with
    a function to initialize its hidden state.

    Args:
        device (str): torch device string, e.g. 'cuda' or 'cpu'.

    Returns:
        model (nn.Module): Loaded, eval-mode dynamics model.
        init_hidden_fn (callable): fn(batch_size) -> hidden state tuple.
    """
    # Load the entire nn.Module
    model = torch.load(MODEL_FILE, map_location=device)
    model.to(device)
    model.eval()

    # Prepare hidden-state initializer
    def init_hidden(batch_size: int = 1):
        # Prefer existing method if defined
        if hasattr(model, 'init_hidden'):
            return model.init_hidden(batch_size)
        # Fallback: look for LSTM attributes
        # Attempt to find any nn.LSTM in model
        for module in model.modules():
            if isinstance(module, torch.nn.LSTM):
                num_layers = module.num_layers
                hidden_size = module.hidden_size
                # return tuple of (h0, c0)
                h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
                c0 = torch.zeros_like(h0)
                return (h0, c0)
        # If no LSTM found, return None or empty tuple
        return None

    return model, init_hidden
