import numpy as np

from .mi import _mi
from .q_cb import _q_cb as Qcb
from .q_g import _q_g as Qg
from .q_mi import _q_mi as Qim
from .q_y import _q_y as Qy


def metrics(l, r, pred):
    lamb = lambda x: np.array(x.squeeze(), dtype=np.float64)
    l, r, pred = lamb(l), lamb(r), lamb(pred)
    return Qim(l, r, pred), Qg(l, r, pred), Qy(l, r, pred), Qcb(l, r, pred)
