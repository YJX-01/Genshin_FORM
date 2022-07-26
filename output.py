import json
from typing import Literal
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np
import squarify
from sim import Sim
from md import *
from strategy import Strategy
from control import Controller


plt.style.use("seaborn-bright")
plt.rcParams['font.sans-serif'] = ['SimHei']


def str_format(n: int) -> str:
    if n < 1000:
        return str(n)
    elif 1000 <= n < 10000:
        return '{:.2f}k'.format(n/1_000)
    elif 10000 <= n < 100000:
        return '{:.2f}w'.format(n/10_000)
    else:
        return '{:.2f}m'.format(n/1_000_000)


def stg_output(ctrl: 'Controller', filename: str = ''):
    'output the strategy'
    containers = [ctrl.Rf, ctrl.Rp, ctrl.Rs, ctrl.Rg, ctrl.Rc]
    pos = {0: '花', 1: '羽', 2: '沙', 3: '杯', 4: '冠'}

    def pie_output(ax, i):
        record = {}
        for j in range(6):
            v = containers[i].stg3.record[str(j)]*0.8 +\
                containers[i].stg4.record[str(j)]*0.2
            if v:
                record[str(j)] = v
        label = [f'+{4*int(k)}' for k in record.keys()]
        size = list(record.values())
        explode = [0 if k != '5' else 0.1 for k in record.keys()]
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap((int(k)+1)/7) for k in record.keys()]
        ax.pie(size, explode=explode, autopct='%1.2f%%',
               startangle=90, colors=colors,
               textprops=dict(color="w"))
        ax.axis('equal')
        ax.set_title(f'{filename} {pos[i]}留弃情况')
        ax.legend(label, loc='upper left')

    def bar_output(ax, i):
        data = {}
        for s, p in containers[i].stg3.desire_dis.items():
            val = ctrl.cnt_s(s, containers[i].stg3.desire_key)
            data.setdefault(val, 0)
            data[val] += p*0.8
        for s, p in containers[i].stg4.desire_dis.items():
            val = ctrl.cnt_s(s, containers[i].stg4.desire_key)
            data.setdefault(val, 0)
            data[val] += p*0.2
        x, y = zip(*sorted(data.items()))
        label = [str(i) for i in range(min(x), max(x)+1)]
        h = np.array(y)/sum(y)
        norm = mpl.colors.Normalize(vmin=0, vmax=len(h)-1)
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(norm(d)) for d in range(len(h))]

        ax.bar(range(len(h)), h, color=colors)
        ax.set_xticks(range(len(h)), labels=label)
        ax.set_xlim(left=-0.5, right=len(h)-0.5)
        ax.grid(axis='y', alpha=0.8)
        ax.yaxis.set_major_formatter('{x:1.1%}')
        ax.set_axisbelow(True)
        ax.set_title(f'{filename} {pos[i]}成品分布情况')
        ax.set_xlabel('有效词条数')
        ax.set_ylabel('概率')

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
    for i, ax_row in enumerate(axs):
        pie_output(ax_row[0], i)
        bar_output(ax_row[1], i)
    plt.tight_layout()
    if filename:
        plt.savefig(f'.\graph\{filename}STG.jpg', dpi=300, format='jpg')
    plt.show()


def loss_output(ctrl: 'Controller', filename: str = ''):
    'output the total loss'
    containers = [ctrl.Rf, ctrl.Rp, ctrl.Rs, ctrl.Rg, ctrl.Rc]
    pos = {0: '花', 1: '羽', 2: '沙', 3: '杯', 4: '冠'}
    loss = []
    for i in range(5):
        g3, l3 = containers[i].stg3.cal_gain(containers[i].f)
        g4, l4 = containers[i].stg4.cal_gain(containers[i].f)
        loss.append(int(l4*0.02))
        loss.append(int(l3*0.08))
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(loss))]
    label = [f'{pos[i//2]}<{4-i%2}>' for i in range(10)]
    plt.figure(figsize=(6, 6))
    squarify.plot(sizes=loss, color=colors, pad=False,
                  label=label, value=loss,
                  text_kwargs=dict(color='w', fontsize='small'))
    plt.text(50, -5, '总代价={}'.format(sum(loss)),
             dict(ha='center', va='center'))
    plt.title(f"{filename} 代价的分布情况")
    plt.axis(False)
    if filename:
        plt.savefig(f'.\graph\{filename}LOSS.jpg', dpi=400, format='jpg')
    plt.show()


def weight_output(ctrl: 'Controller', filename: str = ''):
    'output the weight of each position'
    containers = [ctrl.Rf, ctrl.Rp, ctrl.Rs, ctrl.Rg, ctrl.Rc]
    pos = {0: '花', 1: '羽', 2: '沙', 3: '杯', 4: '冠'}
    eng = ['ATK', 'ATK_PER', 'DEF', 'DEF_PER',
           'HP', 'HP_PER', 'ER', 'EM', 'CR', 'CD']
    ch = ['攻击', '攻击%', '防御', '防御%', '生命', '生命%',
          '元素充能', '元素精通', '暴击率', '暴击伤害']
    map_name = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']

    def bar_output(ax, i):
        w = containers[i].weight
        stat = containers[i].meta_s
        val = [w[eng.index(s)] for s in stat]
        label = [ch[eng.index(s)] for s in stat]
        h = 100*np.array(val)/max(val)  # max = 100
        h = np.rint(h)
        cmap = plt.cm.get_cmap(map_name[i])
        colors = [cmap(j/150) for j in h]
        b = ax.bar(range(len(h)), h, color=colors)
        ax.set_xticks(range(len(h)), labels=label)
        ax.set_xlim(left=-0.5, right=len(h)-0.5)
        ax.bar_label(b, padding=2)
        ax.set_title(f'{filename} {pos[i]}词条权重')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i, ax in enumerate(axs.flat):
        if i == 5:
            ax.axis(False)
            continue
        bar_output(ax, i)
    plt.tight_layout()
    if filename:
        plt.savefig(f'.\graph\{filename}W.jpg', dpi=400, format='jpg')
    plt.show()


def dis_output(ctrl: 'Controller', dis: Dict[tuple, float], filename: str = ''):
    'output the distribution at a specific time'
    # record average stat distribution
    x = sum([np.array(k)*v for k, v in dis.items()])
    stat_dis = dict(zip(ctrl.meta_s, x))
    print(stat_dis)
    with open(f'./data/{filename}DIS.txt', 'w') as f:
        f.writelines([f'{k}: {v:=.2f}\n' for k, v in stat_dis.items()])

    # view value distribution
    val_dis = {}
    mean = 0
    for s, p in dis.items():
        val = ctrl.meta_f(s, ctrl.meta_s)
        val_dis.setdefault(val, 0)
        val_dis[val] += p
        mean += val*p
    x, y = zip(*sorted(val_dis.items()))
    pr = np.cumsum(y)
    left, right = 0, len(pr)
    for i, p in enumerate(pr):
        if p <= 1e-4:
            left = i
        if p <= 1-1e-4:
            right = i
    vs = np.array(x[left:right+1])
    ps = np.array(y[left:right+1])
    interval = max((vs.max()-vs.min())//20, 1)
    div = list(range(vs.min(), vs.max()+1, interval))+[vs.max()+1]
    h, label = [], []
    for i in range(len(div)-1):
        s = sum([ps[j] for j in range(len(vs))
                 if div[i] <= vs[j] < div[i+1]])
        h.append(s)
        n = int(div[i]+interval/2)
        label.append(str_format(n))
    m = (mean-vs.min())*(len(h)-1)/(vs.max()-vs.min())
    print(mean, m, vs.min(), vs.max())
    cmap = plt.cm.get_cmap('viridis')
    colors = [cmap(j/len(h)) for j in range(len(h))]
    mid_c = cmap(0.5)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.bar(range(len(h)), h, width=1, color=colors, edgecolor='snow')
    line1 = ax1.plot([m, m], [0, 1], '--', c='#45488C', label='平均值')
    ax1.annotate(f'{mean:=.1f}', xy=(m, max(h)*1.1),
                 xycoords='data', c='#45488C',
                 xytext=(10, 0), textcoords='offset points')
    ax1.set_xticks(range(len(h)), labels=label, fontsize='x-small')
    ax1.set_xlim(-0.5, len(h)-0.5)
    ax1.set_ylim(0, max(h)*1.2)
    ax1.grid(True, axis='both', alpha=0.6, linestyle=':')
    ax1.set_axisbelow(True)
    ax1.set_title(f'{filename} 数值分布')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('概率', color=mid_c)
    ax1.tick_params(axis='y', labelcolor=mid_c)
    ax1.yaxis.set_major_formatter('{x:1.1%}')

    ax2 = ax1.twinx()
    x = (np.array(x)-vs.min())*(len(h)-1)/(vs.max()-vs.min())
    line2 = ax2.plot(x, pr, color='#ccbb41', label='累积概率')
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.8)
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylabel('累积概率', color='#ccbb41')
    ax2.tick_params(axis='y', labelcolor='#ccbb41')
    ax2.yaxis.set_major_formatter('{x:1.0%}')

    elems = [line2[0], line1[0],
             mpl.patches.Patch(facecolor=mid_c, edgecolor='w', label='分布')]
    plt.legend(handles=elems, loc='upper left')
    fig.tight_layout()
    if filename:
        plt.savefig(f'.\graph\{filename}DIS.jpg', dpi=400, format='jpg')
    plt.show()


def rank_output(ctrl: 'Controller', dis: Dict[tuple, float], filename: str = ''):
    val_dis = {}
    for s, p in dis.items():
        val = ctrl.meta_f(s, ctrl.meta_s)
        val_dis.setdefault(val, 0)
        val_dis[val] += p
    x, y = zip(*sorted(val_dis.items()))
    pr = np.cumsum(y)
    div = list(range(5, 20, 5)) + \
        list(range(20, 80, 2)) + \
        list(range(80, 100, 5))
    rank = dict.fromkeys(div, 0)
    for i, p in enumerate(pr):
        if not div:
            break
        while (div and p > div[0]/100):
            rank[div[0]] = int(x[i])
            div.pop(0)
    print(rank)
    with open(f'./data/{filename}RANK.json', 'w') as f:
        json.dump(rank, f)

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.02)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    cols = list(rank.keys())
    l = np.array(list(rank.values()))
    y = (l-l.min())/(l.max()-l.min())
    min_ratio = int(100*l.min()/l.max())+1
    interval = 3 if min_ratio <= 70 else 1
    ticklabels = [f'{j}%' for j in range(min_ratio, 101, interval)]
    tick = (np.array(range(min_ratio, 101, interval)))*l.max()/100
    tick = (tick-l.min())/(l.max()-l.min())
    ax1.plot(list(range(len(cols))), y, 'go-')
    ax1.set_xlim(-0.5, len(cols)-0.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(tick, labels=ticklabels)
    ax1.set_ylabel('相对最大值的比例')
    ax1.set_xticks(list(range(len(cols))), labels=[])
    ax1.grid(True, 'both')
    ax1.set_title('数值排行表')

    cells = list(rank.values())
    cm = plt.cm.get_cmap('viridis')
    colors = [cm(int(s)/200+0.5) for s in cols]

    ax2.table(cellText=[cells],
              cellColours=[colors],
              rowLabels=[''],
              rowColours=['snow'],
              colLabels=[f'{s}%' for s in cols],
              colColours=colors,
              cellLoc='center',
              bbox=(0, 0.2, 1, 0.8))
    ax2.text(-0.01, 0.6, '排名比例\n\n\n数值', ha='right', va='center')
    ax2.set_axis_off()
    if filename:
        plt.savefig(f'.\graph\{filename}RANK.jpg', dpi=400, format='jpg')
    plt.show()
