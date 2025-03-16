from math import inf

import matplotlib.pyplot as plt
from realkd import rules


def calculate_colours(weights, colours={}):
    max_w = -inf
    min_w = inf
    weights.add(1)
    weights.add(-1)
    weights.add(2)
    weights.add(-2)
    max_w = max(weights)
    min_w = min(weights)
    sorted_weights = sorted(list(weights))
    for i in range(len(sorted_weights)):
        rate = i / (len(sorted_weights) - 1) if len(sorted_weights) > 1 else 1
        colours[sorted_weights[i]] = (rate, 1 - rate, rate)
    return colours


def plot_rules_1d(w, l, u, L, U, titles=['x'], colours={}):
    weights = set()
    for i in range(len(w)):
        weights.add(round(w[i], 6))
    colours = calculate_colours(weights) if len(colours) == 0 else colours
    for j in range(len(titles)):
        lines = {}
        max_x = int(U[j])
        for i in range(len(w)):
            try:
                lines[str(round(u[i][j] - l[i][j], 6)) + str(round(w[i], 6))].append([l[i][j], u[i][j], w[i]])
            except:
                lines[str(round(u[i][j] - l[i][j], 6)) + str(round(w[i], 6))] = [[l[i][j], u[i][j], w[i]]]
        y = 1
        labels = set()
        #     print(lines)
        plt.figure(figsize=(10, len(lines) / 3))
        for line in lines:
            for ln in lines[line]:
                weight = round(ln[2], 6)
                if weight not in labels:
                    plt.hlines(y * 3, ln[0], ln[1], color=colours[weight], lw=10, label=weight)
                    labels.add(weight)
                else:
                    plt.hlines(y * 3, ln[0], ln[1], color=colours[weight], lw=10)
            y += 1

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(titles[j])
        plt.ylim(0, y * 3 + 1)
        plt.xlim(0, max_x)
        ax = plt.gca()
        ax.axes.yaxis.set_ticklabels([])
        plt.grid()
        plt.show()


def plot_steps(steps, data, target):
    colours = {}
    weights = set()

    for step in steps:
        for rule in rules.VisibleRuleEnsemble(step, data).rules:
            weights.add(rule['weight'])
    #     print(weights)
    calculate_colours(weights, colours)
    #     print(colours)
    #     plot_blocks(data, target, colours)
    i = 1
    for step in steps:
        print("Step " + str(i))
        plot_rules_1d(rules.VisibleRuleEnsemble(step, data).rules, x, colours)
        i += 1


def plot_blocks(data, target, colours):
    plotables = []
    plotable = {"start": 0, "length": 0, "weight": 0}
    for i in range(len(target)):
        #         print(data.at[i,'x'])
        if plotable['weight'] != target[i]:
            plotable['start'] = data.at[i, 'x'] + 1
            plotables.append(plotable)
            plotable = {"start": data.at[i, 'x'] + 1, 'length': 0, 'weight': target[i]}
        else:
            plotable['length'] += 1
    plotable['start'] = data.at[i, 'x']
    plotables.append(plotable)
    #     print(plotable)
    print("Dataset")
    plot_rules_1d(plotables[1:], data, colours)