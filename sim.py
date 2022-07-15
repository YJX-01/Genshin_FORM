from typing import Dict, Callable, List
import numpy as np


class Sim(object):
    def __init__(self, f: Callable = None):
        '''
        `value`: function used to evaluate artifact\\
        `origin_dis`: pr for each state\\
        `origin_key`: int value for each state\\
        `trans`: transmission matrix\\
        `rev`: mapping int value to its distribution
        '''
        self.value: Callable[[tuple, list], int] = f
        self.origin_dis: np.ndarray = None
        self.origin_key: List[int] = []
        self.trans: np.ndarray = None
        self.rev: Dict[int, List[tuple]] = None

    def initialize(self, dis: Dict[tuple, float], key: List[str], merge: int = 1):
        '''
        `dis`: desired artifact distribution\n
        `key`: corresponding key of dis\n
        p.s. pr sum doesn't necessarily be 1, the other pr is for undesired(origin_key=0)
        '''
        val_map, rev_map = {}, {}
        val_set = set()
        for point, pr in dis.items():
            val = (self.value(point, key)//merge)*merge
            val_map[point] = val
            val_set.add(val)
            same: list = rev_map.setdefault(val, [(0, 0)])
            if pr > same[0][1]:
                same[0] = (point, pr)
        rev_map[0] = [(tuple([0 for _ in range(len(key))]), 1)]
        self.origin_key = list(sorted(val_set, reverse=True))+[0]
        self.origin_dis = [0 for _ in range(len(val_set))]
        for point, v in val_map.items():
            index = self.origin_key.index(v)
            self.origin_dis[index] += dis[point]
        self.origin_dis.append(1-sum(dis.values()))
        self.origin_dis = np.array(self.origin_dis)
        self.trans = self.to_matrix()
        self.rev = rev_map

    def to_matrix(self):
        l = len(self.origin_dis)
        d = self.origin_dis.copy()
        m = [[0 for i in range(l)] for j in range(l)]
        for i in range(l):
            for j in range(l):
                m[j][l-1-i] = d[j]
            if i < l-1:
                d[l-2-i] += d[l-1-i]
                d[l-1-i] = 0
        return np.array(m)

    def set_value(self, f: Callable):
        self.value = f

    def state_trans(self, times: int) -> np.ndarray:
        '''
        `times`: int, how many times you want to do state transition\n
        `state`: s(0)=[0,...,1] which means begin at no artifact\n
        return s(times)
        '''
        s = np.array([0 for _ in range(len(self.origin_dis)-1)]+[1])
        for i in range(times):
            s = self.trans@s
        return s

    def state_pick(self, points: List[int]) -> Dict[int, List[float]]:
        '''
        `point`: List[int], pick the point at s(n)\n
        return: Dict[key, List[pr]]
        '''
        record = dict.fromkeys(self.origin_key)
        for k in record:
            record[k] = []
        t = 0
        s = np.array([0 for _ in range(len(self.origin_dis)-1)]+[1])
        for p in points:
            for _ in range(p-t):
                s = self.trans@s
            t = p
            for i, k in enumerate(self.origin_key):
                record[k].append(s[i])
        return record

    def state_norm(self, points: List[int]) -> List[tuple]:
        '''return List[Tuple[x, avg, 5%, 25%, 50%, 75%, 95%]]'''
        record = []
        t = 0
        s = np.array([0 for _ in range(len(self.origin_dis)-1)]+[1])
        for p in points:
            for _ in range(p-t):
                s = self.trans@s
            t = p
            avg = sum([s[i]*self.origin_key[i]
                       for i in range(len(s))])
            bound = [-1, -1, -1, -1, -1]
            tmp_sum = 0
            for i in range(len(s)):
                tmp_sum += s[i]
                if bound[0] < 0 and tmp_sum >= 0.05:
                    bound[0] = self.origin_key[i]
                if bound[1] < 0 and tmp_sum >= 0.25:
                    bound[1] = self.origin_key[i]
                if bound[2] < 0 and tmp_sum >= 0.5:
                    bound[2] = self.origin_key[i]
                if bound[3] < 0 and tmp_sum >= 0.75:
                    bound[3] = self.origin_key[i]
                if bound[4] < 0 and tmp_sum >= 0.95:
                    bound[4] = self.origin_key[i]
            record.append((t, avg,
                           bound[0], bound[1], bound[2], bound[3], bound[4]))
        return record
