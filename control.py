import pprint
from typing import Dict, List, Tuple, Callable, Literal
from itertools import combinations
import numpy as np
from artpr import ArtP
from rule import Rule
from sim import Sim
from strategy import Strategy
from view import *


class RuleContainer(object):
    __sub = ['ATK', 'ATK_PER', 'DEF', 'DEF_PER',
             'HP', 'HP_PER', 'ER', 'EM', 'CR', 'CD']

    def __init__(self, pos: str):
        self.pos: str = pos
        self.ms: str = ''
        self.output: bool = True
        self.rule3: 'Rule' = None
        self.rule4: 'Rule' = None
        self.stg3: 'Strategy' = Strategy()
        self.stg4: 'Strategy' = Strategy()
        self.weight: np.ndarray = None
        self.other_x: np.ndarray = None
        self.thresh: Dict[str, Tuple[float, float]] = {}
        self.meta_f: Callable[[tuple, list], int] = None
        self.meta_s: List[str] = None

    def show_all(self):
        print('-'*20, '\n')
        print('position=', self.pos,
              'main stat=', self.ms)
        print('\nweight=',
              ['{:.1f}'.format(w) for w in self.weight], '\n')
        print('strategy_3[{}]:\n'.format(self.pos))
        g3, l3 = self.stg3.cal_gain(self.f)
        print(self.stg3.record, '\n')
        print('strategy_4[{}]:\n'.format(self.pos))
        g4, l4 = self.stg4.cal_gain(self.f)
        print(self.stg4.record, '\n')
        print('x_hat[{}]='.format(self.pos),
              ['{:.2f}'.format(x) for x in self.x_hat], '\n')
        print('total gain[{}]= {:.3f}'.format(self.pos, (g3*0.8+g4*0.2)))
        print('total loss[{}]= {:.0f}\n'.format(self.pos, (l3*0.8+l4*0.2)/10))

    def f(self, s: tuple, name: list) -> int:
        d = dict(zip(name, s))
        r = self.weight[-1]
        for i, n in enumerate(self.__sub):
            x = d.get(n, 0)
            if n in self.thresh:
                t, c = self.thresh[n]
                t -= self.other_x[i]
                if x > t:
                    r += self.weight[i]*(t+(x-t)/np.exp((x-t)/c))
                else:
                    r += self.weight[i]*x
            else:
                r += self.weight[i]*x
        return np.round(max(r, 0))

    def cal_weight(self, x_hat: np.ndarray):
        dp, dy = {}, {}
        s = sum(self.stg3.desire_dis.values())
        for t, p in self.stg3.desire_dis.items():
            x_0 = x_hat+np.array(t)
            y_0 = self.meta_f(x_0)
            x_1 = self.switch_s(x_0)
            dp.setdefault(x_1, 0)
            dp[x_1] += p*0.8/s
            dy[x_1] = y_0
        s = sum(self.stg4.desire_dis.values())
        for t, p in self.stg4.desire_dis.items():
            x_0 = x_hat+np.array(t)
            y_0 = self.meta_f(x_0)
            x_1 = self.switch_s(x_0)
            dp.setdefault(x_1, 0)
            dp[x_1] += p*0.2/s
            dy[x_1] = y_0
        x = np.array(list(dy.keys()))
        y = np.array(list(dy.values()))
        p = np.diag([dp[k] for k in dy.keys()])
        A = np.vstack([x.T, np.ones(len(x))]).T
        Aw = np.dot(p, A)
        yw = np.dot(y, p)
        if self.ms in self.meta_s:
            Aw = np.delete(Aw, self.meta_s.index(self.ms), 1)
        res = np.linalg.lstsq(Aw, yw, rcond=-1)
        s, l = res[0], len(res[0])-1
        if self.ms in self.meta_s:
            s = np.insert(s, self.meta_s.index(self.ms), 0)
        w = []
        for i, n in enumerate(self.__sub):
            if n not in self.meta_s:
                w.append(0)
            else:
                w.append(s[self.meta_s.index(n)])
        # wx+b<0 if stat num |x|<=1 (controversial)
        b = -sum(w)/l
        w.append(b)
        self.weight = np.array(w)
        if self.output:
            print('weight[{}][{}]: '.format(self.pos, self.ms),
                  ['{:.1f}'.format(k) for k in self.weight])
        return self.weight

    @property
    def x_hat(self) -> np.ndarray:
        x = np.zeros(10)
        w = sum(self.stg3.desire_dis.values())
        for t, p in self.stg3.desire_dis.items():
            x += 0.8*(p/w)*np.array(t)
        w = sum(self.stg4.desire_dis.values())
        for t, p in self.stg4.desire_dis.items():
            x += 0.2*(p/w)*np.array(t)
        return x

    @property
    def dis(self) -> Dict[tuple, float]:
        d = {}
        for s, pr in self.stg3.desire_dis.items():
            d[s] = pr*0.8
        for s, pr in self.stg4.desire_dis.items():
            d[s] = pr*0.2
        return d

    @property
    def f_dis(self) -> Dict[tuple, float]:
        d = {}
        for s, pr in self.stg3.desire_dis.items():
            ns = self.switch_s(s)
            d.setdefault(ns, 0)
            d[ns] += pr*0.8
        for s, pr in self.stg4.desire_dis.items():
            ns = self.switch_s(s)
            d.setdefault(ns, 0)
            d[ns] += pr*0.2
        return d

    def switch_s(self, s: tuple) -> tuple:
        return tuple([s[i] for i, n in enumerate(self.__sub)
                      if n in self.meta_s])

    def redo(self, x_hat: np.ndarray, loss3: int, loss4: int):
        self.other_x = x_hat
        self.cal_weight(x_hat)
        self.rule3.initialize(self.pos, self.ms, 3)
        self.rule4.initialize(self.pos, self.ms, 4)
        self.rule3.opt(threshold=loss3, output=self.output)
        self.rule4.opt(threshold=loss4, output=self.output)
        self.stg3.set_rule(self.rule3, self.rule3)
        self.stg4.set_rule(self.rule4, self.rule4)
        self.stg3.analyze(self.pos, self.ms, 3)
        self.stg4.analyze(self.pos, self.ms, 4)
        if self.output:
            print('strategy_3[{}]:\n'.format(self.pos))
            self.stg3.cal_gain(self.f)
            print(self.stg3.record, '\n')
            print('strategy_4[{}]:\n'.format(self.pos))
            self.stg4.cal_gain(self.f)
            print(self.stg4.record, '\n')

    def initialize(self, x_hat: np.ndarray, loss3: int, loss4: int):
        self.other_x = x_hat
        grad = np.zeros(11)
        b = self.meta_f(x_hat)
        for i in range(10):
            tmp = x_hat.copy()
            tmp[i] += 1
            grad[i] = self.meta_f(tmp)-b
        grad[-1] = -sum(grad)/len(self.meta_s)
        self.weight = grad
        self.rule3 = Rule(self.f)
        self.rule4 = Rule(self.f)
        self.rule3.initialize(self.pos, self.ms, 3)
        self.rule4.initialize(self.pos, self.ms, 4)
        self.rule3.opt(threshold=loss3, output=self.output)
        self.rule4.opt(threshold=loss4, output=self.output)
        self.stg3.set_rule(self.rule3, self.rule3)
        self.stg4.set_rule(self.rule4, self.rule4)
        self.stg3.analyze(self.pos, self.ms, 3)
        self.stg4.analyze(self.pos, self.ms, 4)
        if self.output:
            print('strategy_3[{}]:\n'.format(self.pos))
            self.stg3.cal_gain(self.f)
            print(self.stg3.record, '\n')
            print('strategy_4[{}]:\n'.format(self.pos))
            self.stg4.cal_gain(self.f)
            print(self.stg4.record, '\n')


class Controller(object):
    __sub = ['ATK', 'ATK_PER', 'DEF', 'DEF_PER',
             'HP', 'HP_PER', 'ER', 'EM', 'CR', 'CD']

    def __init__(self):
        '''
        you need to set `ms`, `thresh`, `loss`, `meta_f`, `meta_s`, `init_x`
        '''
        self.loss3: Tuple = tuple()
        self.loss4: Tuple = tuple()
        self.init_x: np.ndarray = None

        self.Rf = RuleContainer('flower')
        self.Rp = RuleContainer('plume')
        self.Rs = RuleContainer('sands')
        self.Rg = RuleContainer('goblet')
        self.Rc = RuleContainer('circlet')

        self.thresh: Dict[str, Tuple[float, float]] = {}

        self.meta_f: Callable[[tuple, list], int] = None
        self.meta_s: List[str] = None

    def set_ms(self, sands: str, goblet: str, circlet: str):
        'set the main stat of each position'
        self.Rf.ms = 'HP'
        self.Rp.ms = 'ATK'
        self.Rs.ms = sands
        self.Rg.ms = goblet
        self.Rc.ms = circlet

    def set_thresh(self, stat: str, t: float, c: float):
        '''
        set a threshold on a certain stat(optional)\n
        `t`: threshold value, unit: stat number\n
        `c`: tolerance, unit: stat number (if x>t+c, the gain will decrease)
        '''
        self.thresh[stat] = (t, c)
        self.Rf.thresh = self.Rp.thresh = self.Rs.thresh = \
            self.Rg.thresh = self.Rc.thresh = self.thresh

    def set_f(self, f: Callable[[tuple, list], int]):
        '''set the function `f` you want to optimize (it can be non-linear)'''
        self.meta_f = f
        self.Rf.meta_f = self.Rp.meta_f = self.Rs.meta_f = \
            self.Rg.meta_f = self.Rc.meta_f = self.meta_f

    def set_s(self, s: List[str]):
        'input the variants used in `f`'
        d = dict(zip(self.__sub, range(10)))
        s0 = sorted(s, key=lambda x: d[x])
        self.meta_s = s0
        self.Rf.meta_s = self.Rp.meta_s = self.Rs.meta_s = \
            self.Rg.meta_s = self.Rc.meta_s = self.meta_s

    def set_x(self, x: np.ndarray):
        'set the initial state of each position'
        xs = []
        for i in range(5):
            xs.append(sum([x[j] for j in range(5) if j != i]))
        self.init_x = xs

    def set_loss(self, loss3: tuple, loss4: tuple):
        'set the acceptable loss for each position'
        self.loss3 = loss3
        self.loss4 = loss4

    def do_output(self, flag: bool):
        self.Rf.output = self.Rp.output = self.Rs.output = \
            self.Rg.output = self.Rc.output = flag

    def show_all(self):
        containers = [self.Rf, self.Rp, self.Rs, self.Rg, self.Rc]
        for i in range(5):
            containers[i].show_all()
        # for i in range(5):
            self.justify(containers[i])

    def cnt_s(self, s, name=[]):
        if not name or len(name) == 10:
            return sum(s[i] for i in range(10)
                       if self.__sub[i] in self.meta_s)
        elif name == self.meta_s:
            return sum(s)
        else:
            raise KeyError("stat name is incorrect")

    def info(self):
        return

    def justify(self, r: RuleContainer):
        s_rev = [n for n in self.__sub if n not in self.meta_s]
        c4, c3 = {}, {}
        for l in range(1, min(len(self.meta_s), 4)+1):
            c4.setdefault(l, [0, 0, 0])
            for s1 in combinations(self.meta_s, l):
                for s2 in combinations(s_rev, 4-l):
                    try:
                        b = r.rule4(None, s1+s2, None)
                        c4[l][0] += int(b)
                        c4[l][1] += int(not b)
                        c4[l][2] += 1
                    except:
                        pass
        for l in range(1, min(len(self.meta_s), 3)+1):
            c3.setdefault(l, [0, 0, 0])
            for s1 in combinations(self.meta_s, l):
                for s2 in combinations(s_rev, 3-l):
                    try:
                        b = r.rule3(None, s1+s2, None)
                        c3[l][0] += int(b)
                        c3[l][1] += int(not b)
                        c3[l][2] += 1
                    except:
                        pass
        print('\nposition=', r.pos, 'main stat=', r.ms)
        print('\nchoice of rule4: [stat_num]=[upgrade/abandon/total]:')
        print(c4)
        print('\nchoice of rule3: [stat_num]=[upgrade/abandon/total]:')
        print(c3)

    def product(self, sets: List[Dict]) -> Dict[tuple, float]:
        def join(s1, s2):
            return tuple([s1[i]+s2[i] for i in range(len(s1))])
        s, tmp = sets[0], {}
        for i in range(1, len(sets)):
            tmp = {}
            for s1, p1 in s.items():
                for s2, p2 in sets[i].items():
                    s = join(s1, s2)
                    tmp.setdefault(s, 0)
                    tmp[s] += p1*p2
            s = tmp
            print('len of set=', len(s))
        return s

    def work(self):
        containers = [self.Rf, self.Rp, self.Rs, self.Rg, self.Rc]
        for i in range(5):
            containers[i].initialize(x_hat=self.init_x[i],
                                     loss3=self.loss3[i],
                                     loss4=self.loss4[i])
        if input('do you want to show all?(y/n)') == 'y':
            self.show_all()
        if input('do you want to change loss?(y/n)') == 'y':
            s = input('input loss3, use \'space\' to split each loss')
            self.loss3 = tuple([int(i) for i in s.split()])
            s = input('input loss4, use \'space\' to split each loss')
            self.loss4 = tuple([int(i) for i in s.split()])
        op = 'n'
        while(op != 'y'):
            xs = [containers[i].x_hat for i in range(5)]
            for i in range(5):
                new_x = sum([xs[j] for j in range(5) if j != i])
                print(f'pos={i} ', ['{:.1f}'.format(k) for k in new_x])
                containers[i].redo(new_x, self.loss3[i], self.loss4[i])
            if input('do you want to show all?(y/n)') == 'y':
                self.show_all()
            op = input('do you want to exit?(y/n)')
        for i in range(5):
            containers[i].rule3.complete()
            containers[i].rule4.complete()

    def simulate(self, times: List[int], merge: int = -1) -> Dict[tuple, float]:
        self.Sf = Sim(self.Rf.f)
        self.Sp = Sim(self.Rp.f)
        self.Ss = Sim(self.Rs.f)
        self.Sg = Sim(self.Rg.f)
        self.Sc = Sim(self.Rc.f)
        sims = [self.Sf, self.Sp, self.Ss, self.Sg, self.Sc]
        containers = [self.Rf, self.Rp, self.Rs, self.Rg, self.Rc]
        combs = [containers[i].f_dis for i in range(5)]
        for i in range(5):
            print(f'position {i}, len=', len(combs[i]))
        if merge == -1:
            merge = int(input('do you want to merge the state? merge_num='))

        states = []
        for i in range(5):
            sims[i].initialize(containers[i].f_dis, self.meta_s, merge)
            s = sims[i].state_trans(times[i])
            d = {}
            for j, val in enumerate(sims[i].origin_key):
                if merge:
                    d[sims[i].rev[val][0][0]] = s[j]
                else:
                    pr_sum = sum([t[1] for t in sims[i].rev[val]])
                    for k, pr in sims[i].rev[val]:
                        d[k] = (pr/pr_sum)*s[j]
            states.append(d)
            print('rev len=', len(d))
        final_dis = self.product(states)
        return final_dis


if __name__ == "__main__":
    def func(s, name=[]):
        if len(name) == 5:
            a, aa, em, cr, cd = s
        elif len(name) == 10 or len(name) == 0:
            a, aa, em, cr, cd = s[0], s[1], s[7], s[8], s[9]
        else:
            raise KeyError
        r = (943*(1+0.466+0.496+0.0496*aa)+311+16.54*a) *\
            (1+(0.05+0.2+0.331+0.0331*cr)*(0.5+0.384+0.0662*cd)) *\
            (1+2.78*(19.82*em+80)/(19.82*em+80+1400))*1.5
        return int(r)

    x = np.array([3, 4, 3, 3, 3, 3, 2, 2, 4, 5])

    r = RuleContainer('plume')
    r.ms = 'ATK'
    r.meta_f = func
    r.meta_s = ['ATK', 'ATK_PER', 'EM', 'CR', 'CD']
    r.thresh = {'CR': (6, 4)}

    r.output = False
    r.initialize(x, 40000, 60000)
    r.show_all()
    print('\nweight init:', r.weight)
    r.redo(x, 40000, 60000)
    r.show_all()
    # r.redo(x, 40000, 60000)
    # r.show_all()
    cr = r.x_hat[-2]
    print('\nCR: ', 0.05+0.2+0.331+0.033*(cr+x[-2]))
    print('done')
