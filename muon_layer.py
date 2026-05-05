import torch

class AdamWLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            *args,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            torch.nn.init.constant_(self.bias, 0.)

class AdamWCov1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.init.kaiming_normal_(self.weight)