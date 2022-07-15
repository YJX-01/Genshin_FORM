import pickle
from typing import Dict, Tuple
from collections import OrderedDict


class ArtP(object):
    __main_w = {
        "flower": {"HP": 1000},
        "plume": {"ATK": 1000},
        "sands": {
            "HP_PER": 1334,
            "ATK_PER": 1333,
            "DEF_PER": 1333,
            "ER": 500,
            "EM": 500
        },  # 5000
        "goblet": {
            "HP_PER": 850,
            "ATK_PER": 850,
            "DEF_PER": 800,
            "PRYO_DMG": 200,
            "ELECTRO_DMG": 200,
            "CRYO_DMG": 200,
            "HYDRO_DMG": 200,
            "ANEMO_DMG": 200,
            "GEO_DMG": 200,
            "PHYSICAL_DMG": 200,
            "EM": 100
        },  # 4000
        "circlet": {
            "HP_PER": 1100,
            "ATK_PER": 1100,
            "DEF_PER": 1100,
            "CR": 500,
            "CD": 500,
            "HEAL_BONUS": 500,
            "EM": 200
        }  # 5000
    }

    __sub_w = {
        "ATK": 150,
        "ATK_PER": 100,
        "DEF": 150,
        "DEF_PER": 100,
        "HP": 150,
        "HP_PER": 100,
        "ER": 100,
        "EM": 100,
        "CR": 75,
        "CD": 75
    }  # 1000

    __index = {
        "ATK": 1,
        "ATK_PER": 2,
        "DEF": 3,
        "DEF_PER": 4,
        "HP": 5,
        "HP_PER": 6,
        "ER": 7,
        "EM": 8,
        "CR": 9,
        "CD": 10
    }

    def __init__(self):
        '''
        pr table for substat distribution\\
        position = flower, plume, sands, goblet, circlet\\
        pos_4/pos_3: Dict['position', Dict['main_stat', pr_table]]\n
        sum_{main}(sum(pos_3[position][main].values)) = 1\\
        sum(pos_3[position][main].values) <= 1
        '''
        self.pos_4: Dict[str, Dict] = dict.fromkeys(self.__main_w.keys())
        self.pos_3: Dict[str, Dict] = dict.fromkeys(self.__main_w.keys())
        self.initialize()

    def initialize(self):
        '''
        t_all_3: 3 init stat, main stat != sub stat\\
        t_all_4: 4 init stat, main stat != sub stat\\
        t_3:     3 init stat, {'main_stat': pr_table}\\
        t_4:     4 init stat, {'main_stat': pr_table}
        '''
        t = OrderedDict(self.__sub_w)
        l = []
        for k in t.keys():
            new_t = t.copy()
            new_t.pop(k)
            l.append(new_t)
        self.t_all_4 = self.WS(t, 4)
        self.t_all_3 = self.WS(t, 3)
        self.t_4: Dict[str, Dict] = {}
        self.t_3: Dict[str, Dict] = {}
        for i, k in enumerate(t.keys()):
            self.t_4[k] = self.WS(l[i], 4)
            self.t_3[k] = self.WS(l[i], 3)

        for pos in self.__main_w:
            self.pos_4[pos] = dict.fromkeys(self.__main_w[pos].keys())
            self.pos_3[pos] = dict.fromkeys(self.__main_w[pos].keys())
            weight = sum(self.__main_w[pos].values())
            for main_stat, stat_weight in self.__main_w[pos].items():
                if self.t_4.get(main_stat, {}):
                    tmp_t_4 = {}
                    tmp_t_3 = {}
                    for k, v in self.t_4[main_stat].items():
                        tmp_t_4[k] = v*(stat_weight/weight)
                    for k, v in self.t_3[main_stat].items():
                        tmp_t_3[k] = v*(stat_weight/weight)
                    self.pos_4[pos][main_stat] = tmp_t_4
                    self.pos_3[pos][main_stat] = tmp_t_3
                else:
                    tmp_t_4 = {}
                    tmp_t_3 = {}
                    for k, v in self.t_all_4.items():
                        tmp_t_4[k] = v*(stat_weight/weight)
                    for k, v in self.t_all_3.items():
                        tmp_t_3[k] = v*(stat_weight/weight)
                    self.pos_4[pos][main_stat] = tmp_t_4
                    self.pos_3[pos][main_stat] = tmp_t_3
        return

    @classmethod
    def WS(cls, T: Dict[str, int], n: int) -> Dict[Tuple, float]:
        '''
        weighted sampling from a table for n times\n
        T: {stat: weight} -> pr_table: {(*stats): pr}
        '''
        total_w = sum(T.values())
        pr_table: Dict[Tuple, float] = {}
        for k1, v1 in T.items():
            p1 = v1/total_w
            if n == 1:
                pr_table[(k1,)] = p1
                continue
            next_table = T.copy()
            next_table.pop(k1)
            p = cls.WS(next_table, n-1)
            for k2, v2 in p.items():
                k3 = cls.join(k2, k1)
                v3 = pr_table.setdefault(k3, 0)
                pr_table[k3] = v3 + v2*p1
        return pr_table

    @classmethod
    def join(cls, s: Tuple, a: str) -> Tuple:
        s1 = list(s) + [a]
        s1.sort(key=lambda x: cls.__index[x])
        return tuple(s1)

    def init3_to_4(self, s: Tuple, main_stat: str, p: float = 1) -> Dict[Tuple, float]:
        '''init3 upgrade once and become 4stat'''
        pr_table: Dict[Tuple, float] = {}
        t = OrderedDict(self.__sub_w)
        t.pop(main_stat, 0)
        for k in s:
            t.pop(k)
        weight = sum(t.values())
        for k, v in t.items():
            newkey = self.join(s, k)
            pr_table[newkey] = p*v/weight
        return pr_table


# if __name__ == '__main__':
#     import pprint
    # obj = ArtP()
    # with open('ap.pkl','wb') as f:
    #     pickle.dump(obj, f)

    # with open('ap.pkl','rb') as f:
    #     obj: ArtP = pickle.load(f)
    # pprint.pprint(obj.t_all_4)

# def store2json():
#     import json
#     obj = ArtP()
#     with open('ap.json', 'w') as f:
#         tmp_dict = obj.__dict__
#         for k1, v1 in tmp_dict.items():
#             for k2, v2 in list(v1.items()):
#                 if isinstance(k2, tuple):
#                     v1.pop(k2, None)
#                     k2_ = str(k2)
#                     v1[k2_] = v2
#                 if not isinstance(v2, dict):
#                     continue
#                 for k3, v3 in list(v2.items()):
#                     if isinstance(k3, tuple):
#                         v2.pop(k3, None)
#                         k3_ = str(k3)
#                         v2[k3_] = v3
#                     if not isinstance(v3, dict):
#                         continue
#                     for k4, v4 in list(v3.items()):
#                         if isinstance(k4, tuple):
#                             v3.pop(k4, None)
#                             k4_ = str(k4)
#                             v3[k4_] = v4
#         json.dump(tmp_dict, f, indent=4)
