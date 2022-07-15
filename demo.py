import pickle
import numpy as np
from artpr import ArtP
from control import *


def func(s, name=[]):
    if len(name) == 4:
        a, aa, cr, cd = s
    elif len(name) == 10 or len(name) == 0:
        a, aa, cr, cd = s[0], s[1], s[8], s[9]
    else:
        raise KeyError
    r = (943*(1+0.466+0.496+0.05*aa)+311+16.53*a) *\
        (1+(0.05+0.2+0.331+0.033*cr)*(0.5+0.384+0.066*cd))
    return int(r)


def write():
    ix = np.array([[3, 4, 3, 3, 3, 3, 2, 2, 4, 5],
                   [4, 4, 3, 3, 3, 3, 2, 2, 4, 5],
                   [3, 5, 3, 3, 3, 3, 2, 2, 4, 5],
                   [3, 4, 3, 3, 3, 3, 2, 2, 4, 5],
                   [3, 4, 3, 3, 3, 3, 2, 2, 5, 5]])
    l4 = (60000, 60000, 29000, 9000, 11000)
    l3 = (45000, 45000, 27000, 8000, 9000)

    c = Controller()
    c.set_ms('ATK_PER', 'CRYO_DMG', 'CD')
    c.set_loss(l3, l4)
    c.set_f(func)
    c.set_s(['ATK', 'ATK_PER', 'CR', 'CD'])
    c.set_x(ix)
    c.do_output(False)
    c.work()

    with open('./data/ctrl.pkl', 'wb') as f:
        pickle.dump(c, f)


def load():
    with open('./data/ctrl.pkl', 'rb') as f:
        c: Controller = pickle.load(f)
    return c


if __name__ == "__main__":
    # write()
    c = load()
    c.simulate([40, 40, 40, 80, 40], 'LowSTD')
    c.simulate([200, 200, 200, 400, 200], 'HighSTD')
    c.show_all()
