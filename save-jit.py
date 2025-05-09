import torch 
import lightning

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization

class TBGCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]
    def __init__(
        self,
        encoder_layers: list,
        options: dict = None,
        **kwargs,
    ):
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
        # ======= OPTIONS =======
        options = self.parse_options(options)
        self.cv_normalize = False
        self.cv_min = 0
        self.cv_max = 1
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CV without pre or post/processing modules."""
        
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        
        if self.cv_normalize:
            x = self._map_range(x)
        
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def set_cv_range(self, cv_min, cv_max, cv_std):
        self.cv_normalize = True
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.cv_std = cv_std

    def _map_range(self, x):
        out_max = 1
        out_min = -1
        return (x - self.cv_min) * (out_max - out_min) / (self.cv_max - self.cv_min) + out_min

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_cv(x)
        
        return x
    
class TracedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # lightning module

    def forward(self, x):
        return self.model.encode(x)  # or encode(x)


encoder_layers = [45, 30, 30, 1]
tbgcv = TBGCV(encoder_layers=encoder_layers)

ckpt_path = "./res/0419_185536/model"
ckpt = torch.load(ckpt_path + "/mlcv-final.pt")
tbgcv.load_state_dict(ckpt)

wrapper = TracedWrapper(tbgcv)
example_input = torch.rand(1, 45)
traced = torch.jit.trace(wrapper, example_input)
traced.save(ckpt_path + "/mlcv-jit-final.pt", )