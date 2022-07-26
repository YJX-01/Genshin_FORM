import pickle
from typing import Dict, List, Tuple, Callable, Literal
from itertools import product
import numpy as np
from artpr import ArtP
from rule import Rule
from sim import Sim
from strategy import Strategy
from control import *
from output import *


ch_name = '万叶'
eng_name = 'kazuha'


def func(s, name=[]):
    if len(name) == 2:
        er, em = s
    elif len(name) == 10 or len(name) == 0:
        er, em = s[6], s[7]
    else:
        raise KeyError
    return int(er+em)


def write():
    x = np.array([[0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 0, 0, 0]])
    l4 = (65000, 65000, 30000, 10000, 10000)
    l3 = (50000, 50000, 25000, 7500, 7500)

    print(sum(l4)/10, sum(l3)/10, sum(l4)*0.02+sum(l3)*0.08)

    c = Controller()
    c.set_ms('EM', 'EM', 'EM')
    c.set_loss(l3, l4)
    c.set_f(func)
    c.set_s(['ER', 'EM'])
    c.set_x(x)
    # c.do_output(False)
    c.work()

    with open(f'./data/{eng_name}.pkl', 'wb') as f:
        pickle.dump(c, f)


def load():
    with open(f'./data/{eng_name}.pkl', 'rb') as f:
        c: Controller = pickle.load(f)
    return c


def sim(c: 'Controller'):
    l = c.simulate([40, 40, 40, 80, 40], 0)
    h = c.simulate([200, 200, 200, 400, 200], 0)
    with open(f'./data/{eng_name}LowSTD.pkl', 'wb') as f:
        pickle.dump(l, f)
    with open(f'./data/{eng_name}HighSTD.pkl', 'wb') as f:
        pickle.dump(h, f)


def load_dis():
    with open(f'./data/{eng_name}LowSTD.pkl', 'rb') as f:
        l: Dict[tuple, float] = pickle.load(f)
    with open(f'./data/{eng_name}HighSTD.pkl', 'rb') as f:
        h: Dict[tuple, float] = pickle.load(f)
    return l, h


if __name__ == "__main__":
    # write()
    c = load()
    # sim(c)
    l, h = load_dis()
    # c.show_all()
    # stg_output(c, ch_name)
    # loss_output(c, ch_name)
    # weight_output(c, ch_name)
    # dis_output(c, l, '低-'+ch_name)
    # rank_output(c, l, '低-'+ch_name)
    # dis_output(c, h, '高-'+ch_name)
    # rank_output(c, h, '高-'+ch_name)
