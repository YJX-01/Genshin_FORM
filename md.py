from typing import List, Dict, Tuple, Union, Callable
from collections import OrderedDict
from itertools import product
from math import factorial


class simpleMD(object):
    def __init__(self, keys: List[object] = [], p: List[float] = [], dic: dict = {}):
        '''
        either input (keys, p) or dic, do not input both\n
        - keys: the keys of the events
        - p: the corresponding pr for the events
        - dic = OrderedDict(zip(keys, p))
        '''
        if len(p) != len(keys):
            raise Exception("number not match")
        self.p: List[float] = p
        self.keys: List[object] = keys
        self.dic: OrderedDict[object, float] = OrderedDict()
        if dic:
            self.p = list(dic.values())
            self.keys = list(dic.keys())
        else:
            self.dic = OrderedDict(zip(keys, p))

    def pr(self, events: Union[Dict[object, int], List[int]]) -> float:
        '''return Pr(key1=v1, key2=v2,...)'''
        if isinstance(events, List):
            if len(events) != len(self.p):
                raise Exception("number not match")
            e = events
        elif isinstance(events, Dict):
            e = [events.get(k, 0) for k in self.keys]
        else:
            raise TypeError
        n = sum(e)
        n = factorial(n)
        m = p = 1
        for i in range(len(self.p)):
            m *= factorial(e[i])
            p *= (self.p[i]**e[i])
        return n*p/m


class propagateMD(simpleMD):
    def __init__(self, vectors: List[object] = [], p: List[float] = [], generation: int = 1):
        '''
        input the initial state of the distribution
        - vectors: collections of events as a vector
        - p: the corresponding pr for the events
        - generation: indicate number of choices base on vectors, init = 0, default = 1
        - init_state: default=(1,1,...)
        - pr_table: Dict[point, pr]
        - special: whether it is special case, default=False
        '''
        super().__init__(keys=vectors, p=p)
        self.vectors: List[object] = vectors
        self.generation: int = generation
        self.init_state: tuple = tuple([1 for _ in range(len(p))])
        self.pr_table: Dict[tuple, float] = {}
        self.special: bool = False
        self.generate()

    def generate(self):
        '''generation Pr mapping'''
        iter_seq = [range(self.generation+1) for _ in range(len(self.vectors))]
        for point in product(*iter_seq):
            if sum(point) != self.generation:
                continue
            self.pr_table[point] = self.pr(list(point))
        return

    def propagate(self):
        '''propagate one generation'''
        n = len(self.vectors)
        tmp_table: Dict[tuple, float] = {}
        self.generation += 1
        iter_seq = [range(self.generation+1) for _ in range(n)]
        for point in product(*iter_seq):
            if sum(point) != self.generation:
                continue
            v = 0
            for adjacent, k in self.find_adj(point):
                v += self.pr_table.get(adjacent, 0)*self.p[k]
            if v != 0:
                tmp_table[point] = v
        self.pr_table = tmp_table
        return

    def constrain(self, f: Callable[[tuple, tuple, object], bool], norm: bool = False) -> Dict[tuple, float]:
        '''
        add constraint, and normalize(optional)\n
        f: Callable[[tuple, tuple, object], bool]\\
            (total_point, vectors, md) -> bool\\
        '''
        pr_sum = 0.0
        keep: Dict[tuple, float] = {}
        abandon: Dict[tuple, float] = {}
        for point, v in self.pr_table.items():
            pt = tuple([self.init_state[i]+point[i]
                        for i in range(len(point))])
            if f(pt, self.vectors, self):
                pr_sum += v
                keep[point] = v
            else:
                abandon[point] = v
        if norm:
            for point in keep:
                keep[point] /= pr_sum
        self.pr_table = keep
        return abandon

    @staticmethod
    def find_adj(point: Tuple[int]) -> List[Tuple[tuple, int]]:
        result = []
        for i in range(len(point)):
            if point[i] > 0:
                l = list(point)
                l[i] = l[i] - 1
                result.append((tuple(l), i))
        return result
