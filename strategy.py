import pickle
from typing import Dict, Callable, Tuple, Literal
from collections import Counter
from artpr import ArtP
from md import *


class Strategy(object):
    with open('ap.pkl', 'rb') as f:
        artpr: 'ArtP' = pickle.load(f)

    __sub = ['ATK', 'ATK_PER', 'DEF', 'DEF_PER',
             'HP', 'HP_PER', 'ER', 'EM', 'CR', 'CD']

    cost = {0: 16300, 1: 28425, 2: 42425, 3: 66150, 4: 117175}

    def __init__(self):
        '''
        - `classifier`: Dict['sub_stat', 'desire_lv']\\
        switch sub_stat distribution to desire_lv distribution\n
        - `rule_init`: Callable[[tuple, tuple, object], bool]\\
        it handles ((0,..),(sub_stat_names),md) and decide whether to upgrade initial artifact\n
        - `rules`: Callable[[tuple, tuple, object], bool]
        upgrade strategy at upgrade_time=i. you can refer to self.md to create complex rule\n
        - `record`: record the result distribution
        - `desire_dis`: Dict[`stat`, pr]
        - `desire_key`: List[str], how `stat` is formed
        '''
        self.md: propagateMD = None
        self.classifier: Dict[str, str] = dict(zip(self.__sub, self.__sub))
        self.rule_init: Callable[[tuple, tuple, object], bool] = None
        self.rules: Callable[[tuple, tuple, object], bool] = None
        self.record = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        self.desire_dis: Dict[Tuple, float] = {}
        self.desire_key: List[str] = self.__sub

    def analyze3(self, stat_dis: Dict, main_stat: str) -> Dict[Tuple, float]:
        '''input: pr_table of a specific position and main_stat with init3'''
        self.reset()
        tmp_dis = Counter()
        for point, pr in stat_dis.items():
            if self.rule_init((1, 1, 1), point, None):
                tmp_dis += Counter(self.artpr.init3_to_4(point, main_stat, pr))
            else:
                self.record['0'] += pr
        lv_dis: Dict[Tuple, float] = self.switch_to_lv(tmp_dis)
        for point, pr in lv_dis.items():
            vec, ps, s = [], [], []
            for i in set(point):
                vec.append(i)
                s.append(point.count(i))
                ps.append(0.25*point.count(i))
            self.md = propagateMD(vectors=vec, p=ps, generation=0)  # lv4
            self.md.init_state = tuple(s)
            self.md.special = True  # make num=gen+special correct
            for i in range(1, 5):
                self.record[f'{i}'] += sum(
                    self.md.constrain(self.rules).values())*pr
                self.md.propagate()  # lv8,12,16,20
            self.record['5'] += sum(self.md.pr_table.values())*pr
            for k, v in self.md.pr_table.items():
                newkey = self.to_key(k, self.md)
                self.desire_dis.setdefault(newkey, 0)
                self.desire_dis[newkey] += v*pr
        return self.desire_dis

    def analyze4(self, stat_dis: Dict, main_stat: str = '') -> Dict[Tuple, float]:
        '''input: pr_table of a specific position and main_stat with init4'''
        self.reset()
        tmp_dis = {}
        for point, pr in stat_dis.items():
            if self.rule_init((1, 1, 1, 1), point, None):
                tmp_dis[point] = pr
            else:
                self.record['0'] += pr
        lv_dis: Dict[Tuple, float] = self.switch_to_lv(tmp_dis)
        for point, pr in lv_dis.items():
            vec, ps, s = [], [], []
            for i in set(point):
                vec.append(i)
                s.append(point.count(i))
                ps.append(0.25*point.count(i))
            self.md = propagateMD(vectors=vec, p=ps, generation=1)  # lv4
            self.md.init_state = tuple(s)
            for i in range(1, 5):
                self.record[f'{i}'] += sum(
                    self.md.constrain(self.rules).values())*pr
                self.md.propagate()  # lv8,12,16,20
            self.record['5'] += sum(self.md.pr_table.values())*pr
            for k, v in self.md.pr_table.items():
                newkey = self.to_key(k, self.md)
                self.desire_dis.setdefault(newkey, 0)
                self.desire_dis[newkey] += v*pr
        return self.desire_dis

    def switch_to_lv(self, stat_dis: Dict[Tuple, float]) -> Dict[Tuple, float]:
        '''sub stat distribution -> desire lv distribution'''
        lv_dis: Dict[Tuple, float] = {}
        for sub_stat, pr in stat_dis.items():
            k = tuple(sorted([self.classifier[s] for s in sub_stat]))
            lv_dis.setdefault(k, 0)
            lv_dis[k] += pr
        return lv_dis

    def to_key(self, point: tuple, md: propagateMD) -> tuple:
        l = [0 for _ in range(len(self.desire_key))]
        for i in range(len(md.vectors)):
            l[self.desire_key.index(md.vectors[i])] = point[i]+md.init_state[i]
        return tuple(l)

    def set_classifier(self, classifier: Dict = None):
        if classifier:
            self.classifier = classifier
            self.desire_key = sorted(set(self.classifier.values()))
        else:
            self.classifier = dict(zip(self.__sub, self.__sub))
            self.desire_key = self.__sub

    def set_rule(self, rule_init: Callable, rules: Callable):
        self.rule_init = rule_init
        self.rules = rules

    def reset(self):
        self.record = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        self.desire_dis: Dict[Tuple, float] = {}

    def analyze(self, pos: str, main_stat: str = '', init_num: Literal[3, 4] = 4):
        if init_num == 4:
            dis = self.artpr.pos_4[pos][main_stat]
            return self.analyze4(dis, main_stat)
        else:
            dis = self.artpr.pos_3[pos][main_stat]
            return self.analyze3(dis, main_stat)

    def cal_gain(self, f: Callable) -> Tuple[float, float]:
        'return gain, loss'
        gain = 0
        for s, p in self.desire_dis.items():
            gain += f(s, self.desire_key)*p
        loss = 0
        now = sum(self.record.values())
        for i in range(5):
            now -= self.record[str(i)]
            loss += self.cost[i]*now
        print('R: {:.5e} / {:.1f} = {:.5e}\n'.format(gain, loss, gain/loss))
        return gain, loss
