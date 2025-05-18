import torch 
import lightning

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Transform


def sanitize_range(range: torch.Tensor):
    """Sanitize

    Parameters
    ----------
    range : torch.Tensor
        range to be used for standardization

    """

    if (range < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (range < 1e-6).nonzero(),
        )
    range[range < 1e-6] = 1.0

    return range

class PostProcess(Transform):
    def __init__(
        self,
        stats = None,
        reference_frame_cv = None,
        feature_dim = 1,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("range", torch.ones(feature_dim))
        
        if stats is not None:
            min = stats["min"]
            max = stats["max"]
            self.mean = (max + min) / 2.0
            range = (max - min) / 2.0
            self.range = sanitize_range(range)
        
        if reference_frame_cv is not None:
            self.register_buffer(
                "flip_sign",
                torch.ones(1) * -1 if reference_frame_cv < 0 else torch.ones(1)
            )
        else:
            self.register_buffer("flip_sign", torch.ones(1))
        
    def forward(self, x):
        x = x.sub(self.mean).div(self.range)
        x = x * self.flip_sign
        
        return x


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
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])


# class TBGCV(BaseCV, lightning.LightningModule):
#     BLOCKS = ["norm_in", "encoder",]
#     def __init__(
#         self,
#         encoder_layers: list,
#         options: dict = None,
#         **kwargs,
#     ):
#         super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
#         # ======= OPTIONS =======
#         options = self.parse_options(options)
#         self.cv_normalize = False
#         self.cv_min = 0
#         self.cv_max = 1
        
#         # ======= BLOCKS =======
#         # initialize norm_in
#         o = "norm_in"
#         if (options[o] is not False) and (options[o] is not None):
#             self.norm_in = Normalization(self.in_features, **options[o])

#         # initialize encoder
#         o = "encoder"
#         self.encoder = FeedForward(encoder_layers, **options[o])

#     def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
#         """Evaluate the CV without pre or post/processing modules."""
        
#         if self.norm_in is not None:
#             x = self.norm_in(x)
#         x = self.encoder(x)
        
#         if self.cv_normalize:
#             x = self._map_range(x)
        
#         # x = torch.nn.functional.normalize(x, p=2, dim=1)
#         return x

#     def set_cv_range(self, cv_min, cv_max, cv_std):
#         self.cv_normalize = True
#         self.cv_min = cv_min.detach()
#         self.cv_max = cv_max.detach()
#         self.cv_std = cv_std.detach()

#     def _map_range(self, x):
#         out_max = 1
#         out_min = -1
#         return (x - self.cv_min) * (out_max - out_min) / (self.cv_max - self.cv_min) + out_min