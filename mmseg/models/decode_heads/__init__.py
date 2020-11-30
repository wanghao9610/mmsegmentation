from .aspp_head import ASPPHead
from .da_head import DAHead
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead',
    'DepthwiseSeparableFCNHead'
]
