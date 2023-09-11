from libgptb.losses.jsd import JSD, DebiasedJSD, HardnessJSD
from libgptb.losses.vicreg import VICReg
from libgptb.losses.infonce import InfoNCE, InfoNCESP, DebiasedInfoNCE, HardnessInfoNCE
from libgptb.losses.triplet import TripletMargin, TripletMarginSP
from libgptb.losses.bootstrap import BootstrapLatent
from libgptb.losses.barlow_twins import BarlowTwins
from libgptb.losses.abstract_losses import Loss

__all__ = [
    'Loss',
    'InfoNCE',
    'InfoNCESP',
    'DebiasedInfoNCE',
    'HardnessInfoNCE',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'TripletMarginSP',
    'VICReg',
    'BarlowTwins'
]

classes = __all__
