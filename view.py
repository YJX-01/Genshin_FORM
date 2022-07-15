from typing import Literal
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sim import Sim
from md import *
from strategy import Strategy

plt.style.use("seaborn-bright")


def sum_view(lim: int = 5, vec=[1, 1, 1]):
    vsum = lim - 1
    vector = np.array(vec)
    x, y, z = np.indices((lim, lim, lim))

    constraints1 = (x+y+z) == vsum

    shape = constraints1

    values = np.empty(shape.shape)

    for i, j, k in product(range(lim), range(lim), range(lim)):
        if shape[i][j][k]:
            v = np.inner(vector, np.array([i, j, k]))
            if vsum == 0 or vector.max() == vector.min():
                values[i][j][k] = 0
                continue
            values[i][j][k] = (v/vsum-vector.min()) / \
                (vector.max()-vector.min())

    fc = plt.colormaps['coolwarm'](values)

    ax = plt.figure(dpi=200).add_subplot(projection='3d')
    ax.voxels(shape, facecolors=fc, edgecolor='none', alpha=0.95)
    ax.set(xlabel='x', ylabel='y', zlabel='z', title=f'lim={lim}')
    ax.set_xticks(np.arange(lim+1))
    ax.set_yticks(np.arange(lim+1))
    ax.set_zticks(np.arange(lim+1))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.view_init(45, 45)

    plt.show()


def pr_view(lim: int = 5, p=[]):
    vsum = lim - 1
    p = np.ones(3)/3 if not p else p
    x, y, z = np.indices((lim, lim, lim))

    constraints1 = (x+y+z) == vsum

    shape = constraints1

    values = np.empty(shape.shape)

    M = simpleMD(p=p, keys=['x', 'y', 'z'])
    l, h = 1, 0
    for i, j, k in product(range(lim), range(lim), range(lim)):
        if shape[i][j][k]:
            if vsum == 0:
                values[i][j][k] = 0
                continue
            v = M.pr([i, j, k])
            l = min(l, v)
            h = max(h, v)
            values[i][j][k] = v

    if h == l:
        values = np.empty(shape.shape)
    else:
        values = (values-l)/(h-l)

    fc = plt.colormaps['coolwarm'](values)

    ax = plt.figure(dpi=200).add_subplot(projection='3d')
    ax.voxels(shape, facecolors=fc, edgecolor='none', alpha=0.95)
    ax.set(xlabel='x', ylabel='y', zlabel='z', title=f'lim={lim}')
    ax.set_xticks(np.arange(lim+1))
    ax.set_yticks(np.arange(lim+1))
    ax.set_zticks(np.arange(lim+1))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.view_init(45, 45)

    plt.show()


def md_view(md: propagateMD, c: Literal['norm', 'abs'] = 'norm'):
    '''only the first three dimension will be shown'''
    lim = md.generation+1
    xlabel, ylabel, zlabel = str(md.vectors[0]), str(
        md.vectors[1]), str(md.vectors[2])
    x, y, z = np.indices((lim, lim, lim))
    l, h = 1, 0
    shape = np.full(x.shape, False)
    values = np.zeros(shape.shape)

    for point, v in md.pr_table.items():
        shape[point[0]][point[1]][point[2]] = True
        values[point[0]][point[1]][point[2]] = v
        l = min(l, v)
        h = max(h, v)

    if c == 'norm':
        if h == l:
            values = np.zeros(shape.shape)
        else:
            values = (values-l)/(h-l)
        fc = plt.colormaps['coolwarm'](values)
    elif c == 'abs':
        values = np.power(values, (1/4))
        fc = plt.colormaps['coolwarm'](values)

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection='3d')
    ax.voxels(shape, facecolors=fc, edgecolor='none', alpha=0.95)
    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
           title=f'upgrade times={lim-1}')
    ax.set_xticks(np.arange(lim+1))
    ax.set_yticks(np.arange(lim+1))
    ax.set_zticks(np.arange(lim+1))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.view_init(45, 45)

    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(l, h), cmap=plt.cm.coolwarm),
                 ax=ax, label='Normalize Pr', format='%.3f', ticks=np.linspace(l, h, 11))
    plt.show()


def sim_view(sim: Sim, low: int, high: int, interval: int, filename: str = ''):
    fig, ax = plt.subplots(figsize=(int((high-low)/16), 10))
    xt = list(range(low, high+1, interval))
    record = sim.state_pick(xt)
    norm = mpl.colors.Normalize(
        vmin=sim.origin_key[-2]-1, vmax=sim.origin_key[0], clip=True)
    cmap = plt.cm.get_cmap('viridis')
    colors = [cmap(norm(s)) for s in record]
    ax.stackplot(xt, record.values(), labels=record.keys(),
                 alpha=0.8, colors=colors)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1),
              ncol=5, fontsize='xx-small')
    ax.set_title(f'{filename} State transition',
                 fontdict=dict(fontsize='xx-large'))
    ax.set_xlabel('times')
    ax.set_ylabel('Probability')
    ax.set_xlim(left=low, right=high)
    ax.set_ylim(top=1, bottom=0)
    if interval >= 5:
        ax.set_xticks(xt, fontsize='xx-small')
    ax.set_yticks(np.linspace(0, 1, 11))
    if filename:
        plt.savefig(f'.\graph\{filename}Sim.jpg', dpi=250, format='jpg')
    plt.show()


def bound_view(sim: Sim, low: int, high: int, interval: int, filename: str = ''):
    fig, ax = plt.subplots(figsize=(int((high-low)/16), 10))
    xt = list(range(low, high+1, interval))
    record = sim.state_norm(xt)
    l = len(xt)
    ax.plot(xt, [record[i][1] for i in range(l)],
            label='avg', c='darkviolet', linewidth=4)
    ax.plot(xt, [record[i][2] for i in range(l)],
            ':', label='5%', c='darkred', linewidth=2, alpha=0.8)
    ax.plot(xt, [record[i][3] for i in range(l)],
            '--', label='25%', c='red', linewidth=2, alpha=0.8)
    ax.plot(xt, [record[i][4] for i in range(l)],
            '-', label='50%', c='dimgrey', linewidth=3, alpha=0.9)
    ax.plot(xt, [record[i][5] for i in range(l)],
            '--', label='75%', c='blue', linewidth=2, alpha=0.8)
    ax.plot(xt, [record[i][6] for i in range(l)],
            ':', label='95%', c='darkblue', linewidth=2, alpha=0.8)
    ax.set_title(f'{filename} Mean and Variance',
                 fontdict=dict(fontsize='xx-large'))
    ax.set_xlabel('times')
    ax.set_ylabel('value')
    ax.set_xlim(left=low, right=high)
    ax.set_ylim(bottom=0, top=max(sim.origin_key))
    if interval >= 5:
        ax.set_xticks(xt, fontsize='xx-small')
    ax.grid(axis='both')
    plt.legend(loc='lower right', fontsize='large')
    if filename:
        plt.savefig(f'.\graph\{filename}Bound.jpg', dpi=300, format='jpg')
    plt.show()


def str_format(n: int) -> str:
    if n < 1000:
        return str(n)
    elif 1000 <= n < 10000:
        return '{:.2f}k'.format(n/1_000)
    elif 10000 <= n < 100000:
        return '{:.2f}w'.format(n/10_000)
    else:
        return '{:.2f}m'.format(n/1_000_000)


def stg_view(stg: Strategy, f: Callable[[tuple, list], int], filename: str = ''):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    label = list(stg.record.keys())
    size = list(stg.record.values())
    explode = [0 for _ in range(len(size))]
    explode[label.index('5')] = 0.1
    cmap = plt.cm.get_cmap('viridis')
    cl = [cmap((i+1)/7) for i in range(6)]
    ax1.pie(size, explode=explode, autopct='%1.3f%%',
            startangle=90, colors=cl,
            textprops=dict(color="w"))
    ax1.axis('equal')
    ax1.set_title(f'{filename} Abandoned Artifact Ratio')
    ax1.legend(label, loc='upper left')

    data = {}
    for s, p in stg.desire_dis.items():
        val = f(s, stg.desire_key)
        data.setdefault(val, 0)
        data[val] += p
    xl, h = zip(*data.items())
    lb = [str(i) for i in range(min(xl), max(xl)+1)]
    if len(xl) > 20 or max(xl)-min(xl) > 30:
        interval = (max(xl)-min(xl))//15
        div = list(range(min(xl), max(xl), interval))+[max(xl)+1]
        tmp, lb = [], []
        for i in range(len(div)-1):
            s = sum([h[j] for j in range(len(h))
                     if div[i] <= xl[j] < div[i+1]])
            tmp.append((div[i], s))
            lb.append(str_format(int(div[i]+interval/2)))
        xl, h = zip(*tmp)
    norm = mpl.colors.Normalize(vmin=min(xl), vmax=max(xl))
    cl = [cmap(norm(d)) for d in xl]

    ax2.bar(range(len(xl)), h, color=cl)
    ax2.set_xticks(range(len(xl)), labels=lb)
    ax2.set_xlim(left=-0.5, right=len(xl))
    ax2.grid(axis='y', alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title(f'{filename} Finished Artifact Distribution')
    ax2.set_xlabel('value')
    ax2.set_ylabel('Pr')
    if filename:
        plt.savefig(f'.\graph\{filename}Strategy.jpg', dpi=400, format='jpg')
    plt.show()


def val_view(dis: Dict[tuple, float], key: List[str], f: Callable[[tuple, list], int], filename: str = ''):
    val_dis = {}
    mean = 0
    for s, p in dis.items():
        val = f(s, key)
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
    interval = (vs.max()-vs.min())//20
    m = (mean-vs.min()-interval/2)/interval
    div = list(range(vs.min(), vs.max(), interval))+[vs.max()+1]
    h, label = [], []
    for i in range(len(div)-1):
        s = sum([ps[j] for j in range(len(vs))
                 if div[i] <= vs[j] < div[i+1]])
        h.append(s)
        n = int(div[i]+interval/2)
        label.append(str_format(n))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(len(h)), h, width=1, color='royalblue', edgecolor='snow')
    line1 = ax1.plot([m, m], [0, 1], '--', c='turquoise', label='mean')
    ax1.annotate('{:.0f}'.format(mean), xy=(m, max(h)*1.1),
                 xycoords='data', c='turquoise',
                 xytext=(10, 0), textcoords='offset points')
    ax1.set_xticks(range(len(h)), labels=label, fontsize='x-small')
    ax1.set_xlim(-0.5, len(h)-0.5)
    ax1.set_ylim(0, max(h)*1.2)
    ax1.grid(True, axis='both', alpha=0.6, linestyle=':')
    ax1.set_axisbelow(True)
    ax1.set_title(f'{filename} value distribution')
    ax1.set_xlabel('value')
    ax1.set_ylabel('pr', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    ax2 = ax1.twinx()
    x = (np.array(x)-vs.min()-interval/2)/interval
    line2 = ax2.plot(x, pr, color='coral', alpha=0.8, label='cumulative pr')
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.8)
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylabel('cpr', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    elems = [line2[0], line1[0],
             mpl.patches.Patch(facecolor='royalblue', edgecolor='snow', label='distribution')]
    plt.legend(handles=elems, loc='upper left')
    fig.tight_layout()
    if filename:
        plt.savefig(f'.\graph\{filename}ValueDis.jpg', dpi=600, format='jpg')
    plt.show()
