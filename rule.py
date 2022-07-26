import pickle
from typing import Dict, List, Tuple, Callable, Literal
import numpy as np
from artpr import ArtP
from md import *


class Rule(object):
    with open('ap.pkl', 'rb') as f:
        artpr: 'ArtP' = pickle.load(f)

    __sub = ['ATK', 'ATK_PER', 'DEF', 'DEF_PER',
             'HP', 'HP_PER', 'ER', 'EM', 'CR', 'CD']

    cost = {0: 16300, 1: 28425, 2: 42425, 3: 66150, 4: 117175}

    def __init__(self, f: Callable[[tuple, list], float]):
        self.value: Callable[[tuple, list], float] = f
        self.spaces: List[Dict[tuple, int]] = []   # 0,1,2,3,4,5
        self.pr: List[np.ndarray] = []             # 0,1,2,3,4,5
        self.trans: List[np.ndarray] = []          # 0,1,2,3,4
        self.choices: List[np.ndarray] = []        # 0,1,2,3,4
        self.omega: np.ndarray = None

        self.pos: str = ''
        self.main_stat: str = ''
        self.init_num: Literal[3, 4] = 4

    def initialize(self, pos: str, main_stat: str = '', init_num: Literal[3, 4] = 4):
        self.reset()
        self.pos, self.main_stat, self.init_num = pos, main_stat, init_num
        if init_num == 4:
            dis = self.artpr.pos_4[pos][main_stat]
            self.init4(dis)
        else:
            dis = self.artpr.pos_3[pos][main_stat]
            self.init3(dis, main_stat)

    def init4(self, init_dis: Dict[tuple, float]):
        states = [self.to_key(s) for s in init_dis.keys()]
        self.spaces.append(dict(zip(states, range(len(states)))))
        for i in range(5):
            next_states = set()
            for s in states:
                for ns in self.find_adj(s):
                    next_states.add(ns)
            states = list(next_states)
            self.spaces.append(dict(zip(states, range(len(states)))))

        for i in range(5):
            state_id = self.spaces[i]
            next_state_id = self.spaces[i+1]
            tran = [None for _ in range(len(state_id))]
            for s, index in state_id.items():
                d = np.zeros(len(next_state_id))
                for ns in self.find_adj(s):
                    d[next_state_id[ns]] = 0.25
                tran[index] = d
            self.trans.append(np.array(tran))

        prob = np.zeros(len(self.spaces[0]))
        for s, p in init_dis.items():
            prob[self.spaces[0].get(self.to_key(s))] = p
        self.pr.append(prob)
        for i in range(5):
            self.choices.append(np.ones(len(prob), dtype=bool))
            A = np.array(self.trans[i])
            next_prob = prob @ A
            prob = next_prob
            self.pr.append(prob)

        end_space = self.spaces[5]
        self.omega = np.zeros(len(end_space))
        for s, index in end_space.items():
            self.omega[index] = self.value(s, self.__sub)
        return

    def init3(self, init_dis, main_stat):
        states = [self.to_key(s) for s in init_dis.keys()]
        self.spaces.append(dict(zip(states, range(len(states)))))

        next_states = set()
        tran = [None for _ in range(len(self.spaces[0]))]
        tran_d = {}
        for s in init_dis.keys():
            dic = self.artpr.init3_to_4(s, main_stat)
            index = self.spaces[0][self.to_key(s)]
            tran_d[index] = dic
            for ns in dic.keys():
                next_states.add(self.to_key(ns))
        states = list(next_states)
        self.spaces.append(dict(zip(states, range(len(states)))))

        state_id = self.spaces[1]
        for index, dic in tran_d.items():
            d = np.zeros(len(state_id))
            for ns, p in dic.items():
                d[state_id[self.to_key(ns)]] = p
            tran[index] = d
        self.trans.append(np.array(tran))

        for i in range(1, 5):
            next_states = set()
            for s in states:
                for ns in self.find_adj(s):
                    next_states.add(ns)
            states = list(next_states)
            self.spaces.append(dict(zip(states, range(len(states)))))

        for i in range(1, 5):
            state_id = self.spaces[i]
            next_state_id = self.spaces[i+1]
            tran = [None for _ in range(len(state_id))]
            for s, index in state_id.items():
                d = np.zeros(len(next_state_id))
                for ns in self.find_adj(s):
                    d[next_state_id[ns]] = 0.25
                tran[index] = d
            self.trans.append(np.array(tran))

        prob = np.zeros(len(self.spaces[0]))
        for s, p in init_dis.items():
            prob[self.spaces[0].get(self.to_key(s))] = p
        self.pr.append(prob)
        for i in range(5):
            self.choices.append(np.ones(len(prob), dtype=bool))
            A = np.array(self.trans[i])
            next_prob = prob @ A
            prob = next_prob
            self.pr.append(prob)

        end_space = self.spaces[5]
        self.omega = np.zeros(len(end_space))
        for s, index in end_space.items():
            self.omega[index] = self.value(s, self.__sub)
        return

    @staticmethod
    def find_adj(s: Tuple[int]) -> List[Tuple[int]]:
        result = []
        for i in range(len(s)):
            if s[i] > 0:
                l = list(s)
                l[i] += 1
                result.append(tuple(l))
        return result

    def to_key(self, s: Tuple[str], v: Tuple[int] = None) -> Tuple[int]:
        l = []
        for name in self.__sub:
            try:
                i = s.index(name)
                t = 1 if not v else v[i]
                l.append(t)
            except ValueError:
                l.append(0)
        return tuple(l)

    def reset(self):
        self.spaces = []
        self.pr = []
        self.trans = []
        self.choices = []
        self.omega = None

    def opt(self, iter: int = 1, threshold: float = 300000, output: bool = False):
        self.threshold = threshold
        choice = (0, 0, 0, 0, 0)
        for round in range(iter):
            for gen in range(4, -1, -1):
                A = (self.trans[4].T*self.choices[4])
                for i in range(3, gen-1, -1):
                    A = A@(self.trans[i].T*self.choices[i])

                w = self.omega@A
                y = sum([np.matmul(self.pr[g], self.choices[g])*self.cost[g]
                         for g in range(5) if g < gen])

                p = self.to_p(gen)
                self.choices[gen] = np.zeros(len(w), dtype=bool)
                self.find_k(y, w, p, gen)

                for g in range(gen, 5, 1):
                    self.pr[g+1] = self.pr[g] @ \
                        (self.trans[g].T*self.choices[g]).T
                    if g+1 < 5:
                        self.choices[g+1] = self.choices[g+1]\
                            & (self.pr[g+1] > 0)

            choice_tmp = tuple([sum(self.choices[i]) for i in range(5)])
            if output:
                for i in range(5):
                    a, b = sum(self.choices[i]), len(self.choices[i])
                    print(f'at {i} gen:',
                        'up={:<4}, total={:<4}, ratio={:.3%},'.format(a, b, a/b),
                        'pr={:.3%}'.format(sum(self.pr[i])))
                print(choice_tmp)
            if choice_tmp == choice:
                break
            else:
                choice = choice_tmp

    def find_k(self, y, w, p, gen):
        order = sorted(zip(w/p, range(len(w))), reverse=True)
        pr = self.pr[gen]
        upper, lower = 0, y
        for v, index in order:
            if w[index] <= 0:
                break
            if upper*p[index] > lower*w[index] and lower > self.threshold:
                break
            else:
                upper += w[index]*pr[index]
                lower += p[index]*pr[index]
                if pr[index] > 0:
                    self.choices[gen][index] = True

    def to_p(self, gen: int) -> np.ndarray:
        n = len(self.spaces[gen])
        p = np.zeros(n)
        A = None
        for i in range(0, 5-gen, 1):
            if i == 0:
                p += self.cost[gen]*np.ones(n)
            elif i == 1:
                A = self.trans[gen].T
                p += self.cost[gen+1]*(self.choices[gen+1]@A)
            else:
                t = gen+i-1
                A = (self.trans[t].T*self.choices[t])@A
                p += self.cost[gen+i]*(self.choices[gen+i]@A)
        return p

    def complete(self):
        self.pr.clear()
        self.trans.clear()
        self.omega = None

    def __call__(self, point: tuple, vector: tuple, md: object):
        k = self.to_key(vector, point)
        g = sum(point)-4+int(md.special) if md else 0
        id = self.spaces[g][k]
        return bool(self.choices[g][id])
