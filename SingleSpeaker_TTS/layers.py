import torch
import torch.nn as nn
import torch.nn.functional as F
#TACOTRON2 MODEL IMPLEMENTATION

#masking
def get_mask_from_lengths(lengths, max_len = None):
    """
    lengths: LongTensor [B]
    returns: BoolTensor mask [B, max_len] where True indicates padding positions
    """
    B = lengths.size(0)
    if max_len is None:
        max_len=lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    mask=ids>=lengths.unsqueeze(1)
    return mask

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias = False, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        #xavier initialization
        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )
    def forward(self, x):
            return self.linear(x)