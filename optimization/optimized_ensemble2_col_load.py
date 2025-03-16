from math import exp, log

import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd
from build_rule_ensemble import build_ensemble


def squared_loss(yy, y):
    return (yy - y) * (yy - y)


def optimized_rule_ensemble2(n, d, k, L, U, x, y, labels, reg=0, loss_func='squared', tl=100,
                             init_w=[], init_l=[], init_u=[], f=None, debug=False, max_col_num=10):
    new_x = []
    for i in range(len(x)):
        new_x.append([i] + list(x[i]))
    # print('new x', new_x)
    m = gp.Model("rule1")
    log_loss_y = [4.0181499279178094, 3.048587351573742, 2.1269280110429727, 1.3132616875182228,
                  0.6931471805599453, 0.31326168751822286, 0.1269280110429726, 0.04858735157374196,
                  0.01814992791780978, 0, 0]
    m.setParam(GRB.Param.TimeLimit, tl)
    if loss_func != 'squared':
        m.setParam(GRB.Param.NonConvex, 2)
    weight = m.addVars(k, vtype=GRB.CONTINUOUS, lb=-500000, name="weight")
    s = m.addVars(n, k, d, vtype=GRB.BINARY, name="s")
    z = m.addVars(n, k, vtype=GRB.BINARY, name="z")
    w = m.addVars(n, k, vtype=GRB.CONTINUOUS, lb=-500000, name="w")
    yy = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-500000, name="yy")
    loss = m.addVars(n, vtype=GRB.CONTINUOUS, name="loss")
    mult = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-500000, name="mult")
    obj = m.addVar(name='obj', vtype=GRB.CONTINUOUS)
    t = m.addVars(n, k, d, vtype=GRB.BINARY, name='t')
    u = m.addVars(n, k, d, vtype=GRB.BINARY, name='u')
    m.update()
    if len(init_w) != 0:
        for r in range(k):
            weight[r].start = init_w[r]
    if len(init_l) != 0 and len(init_u) != 0:
        for i in range(n):
            for r in range(k):
                for j in range(d):
                    s[i, r, j].start = 1 if init_l[r][j] <= x[i][j] <= init_u[r][j] else 0
                    t[i, r, j].start = 1 if init_l[r][j] <= x[i][j] else 0
                    u[i, r, j].start = 1 if init_u[r][j] >= x[i][j] else 0
    for j in range(d):
        temp_x = sorted(new_x, key=lambda row: row[j + 1])
        interval_len = (U[j] - L[j]) / (max_col_num)
        for r in range(k):
            for ii in range(max_col_num):
                for i in range(n):
                    if i + 1 < n and ii * interval_len + L[j] <= temp_x[i][j + 1] < (
                            ii + 1) * interval_len + L[j] and ii * interval_len + L[j] <= \
                            temp_x[i + 1][j + 1] < (ii + 1) * interval_len + L[j]:
                        m.addConstr(t[temp_x[i][0], r, j] == t[temp_x[i + 1][0], r, j])
                        m.addConstr(u[temp_x[i][0], r, j] == u[temp_x[i + 1][0], r, j])
                    if temp_x[i][j + 1] >= (ii + 1) * interval_len + L[j]:
                        break
            m.addConstr(gp.quicksum(s[i, r, j] for i in range(n)) >= 1, name='sum_s_ge_1_' + str(j) + str(r))
            for i in range(n):
                if i == 0:
                    if temp_x[i][j + 1] == temp_x[i + 1][j + 1]:
                        m.addConstr(t[temp_x[i][0], r, j] ==
                                    t[temp_x[i + 1][0], r, j])
                        m.addConstr(u[temp_x[i][0], r, j] ==
                                    u[temp_x[i + 1][0], r, j])
                    m.addConstr(t[temp_x[i][0], r, j] <=
                                t[temp_x[i + 1][0], r, j])
                    m.addConstr(u[temp_x[i][0], r, j] == 1)
                elif i == n - 1:
                    if temp_x[i][j + 1] == temp_x[i - 1][j + 1]:
                        m.addConstr(t[temp_x[i][0], r, j] ==
                                    t[temp_x[i - 1][0], r, j])
                        m.addConstr(u[temp_x[i][0], r, j] ==
                                    u[temp_x[i - 1][0], r, j])
                    m.addConstr(t[temp_x[i][0], r, j] == 1)
                    m.addConstr(u[temp_x[i][0], r, j] <=
                                u[temp_x[i - 1][0], r, j])
                else:
                    if temp_x[i][j + 1] == temp_x[i - 1][j + 1]:
                        m.addConstr(u[temp_x[i][0], r, j] ==
                                    u[temp_x[i - 1][0], r, j])
                        m.addConstr(t[temp_x[i][0], r, j] ==
                                    t[temp_x[i - 1][0], r, j])
                    m.addConstr(t[temp_x[i][0], r, j] >=
                                t[temp_x[i - 1][0], r, j])
                    m.addConstr(u[temp_x[i][0], r, j] <=
                                u[temp_x[i - 1][0], r, j])
                m.addConstr(s[temp_x[i][0], r, j] == t[temp_x[i]
                                                       [0], r, j] + u[temp_x[i][0], r, j] - 1)
                m.addConstr(z[i, r] <= s[i, r, j], name='z1-' + str(r) + str(j) + str(i))
    for r in range(k):
        for i in range(n):
            m.addConstr(z[i, r] >= gp.quicksum(s[i, r, j] for j in range(d)) - d + 1, name='z2-' + str(r) + str(i))
            m.addConstr((z[i, r] == 1) >> (w[i, r] == weight[r]), name='w' + str(r) + str(i))
            m.addConstr((z[i, r] == 0) >> (w[i, r] == 0))
            # m.addConstr(w[i, r] == z[i, r] * weight[r])
    for i in range(n):
        m.addConstr(yy[i] == gp.quicksum(w[i, r] for r in range(k)), name='yy' + str(i))
        if loss_func == 'logistic':
            m.addConstr(mult[i] == yy[i] * y[i], name='multi' + str(i))
            m.addGenConstrPWL(mult[i], loss[i], [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], log_loss_y, 'log_loss')
    if loss_func == "squared":
        m.setObjective((gp.quicksum((yy[i] - y[i]) * (yy[i] - y[i]) for i in range(n)) + reg * gp.quicksum(
            weight[i] * weight[i] for i in range(k)) / 2) / n)
    elif loss_func == 'logistic':
        m.addConstr(obj == (gp.quicksum(loss[i] for i in range(n)) + reg * gp.quicksum(
            weight[i] * weight[i] for i in range(k)) / 2) / n, name='log_loss')
        m.setObjective(obj)
    m.optimize()
    if debug:
        for j in range(d):
            sss = []
            ttt = []
            uuu = []
            temp_x = sorted(new_x, key=lambda row: row[j + 1])
            print(temp_x)
            for r in range(k):
                ss = []
                tt = []
                uu = []
                for i in range(n):
                    ss.append(s[temp_x[i][0], r, j].x)
                    tt.append(t[temp_x[i][0], r, j].x)
                    uu.append(u[temp_x[i][0], r, j].x)
                sss.append(ss)
                ttt.append(tt)
                uuu.append(uu)
            print('====== s =======')
            for ss in sss:
                print(ss)
            print('====== t =======')
            for tt in ttt:
                print(tt)
            print('====== u =======')
            for uu in uuu:
                print(uu)
    res_lowers = [[l for l in L] for _ in range(k)]
    res_uppers = [[u for u in U] for _ in range(k)]
    for j in range(d):
        temp_x = sorted(new_x, key=lambda row: row[j + 1])
        for r in range(k):
            for i in range(n):
                if i == 0:
                    if abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6:
                        res_lowers[r][j] = temp_x[i][j + 1]
                    if abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6 and abs(s[temp_x[i + 1][0], r, j].x) <= 1e-6:
                        res_uppers[r][j] = temp_x[i][j + 1]
                elif i == n - 1:
                    if abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6:
                        res_uppers[r][j] = temp_x[i][j + 1]
                    if abs(s[temp_x[i - 1][0], r, j].x) <= 1e-6 and abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6:
                        res_lowers[r][j] = temp_x[i][j + 1]
                else:
                    if abs(s[temp_x[i - 1][0], r, j].x) <= 1e-6 and abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6:
                        res_lowers[r][j] = temp_x[i][j + 1]
                    if abs(s[temp_x[i][0], r, j].x - 1) <= 1e-6 and abs(s[temp_x[i + 1][0], r, j].x) <= 1e-6:
                        res_uppers[r][j] = temp_x[i][j + 1]

    return res_lowers, res_uppers, [weight[wi].x for wi in range(k)], [yy[yi].X for yi in
                                                                       yy], m.getObjective().getValue(), m.ObjBound


if __name__ == '__main1__':
    labels = ['GDP']
    train, test, train_target, test_target, L, U, d, n = preprocess_pd(
        '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
        labels, [int], 'Satisfaction', target_type=float)
    print(train)
    print(train_target)
    l, u, w, yy, obj, bnd = optimized_rule_ensemble2(n, d, 10, L, U, train, train_target, ['GDP'], tl=5)
    sum1 = 0
    sum2 = 0
    print(w)
    print(l)
    print(u)
    print(yy)
    ensemble = build_ensemble(w, l, u, L, U, ['GDP'])
    train_df = pd.DataFrame(train, columns=labels)
    yyy = ensemble(train_df)
    print(yyy)
    for i in range(len(train)):
        sum1 += (yy[i] - train_target[i]) ** 2
        sum2 += (yyy[i] - train_target[i]) ** 2
    print(sum1 / len(yy), sum2 / len(yyy))


def squared_loss(y, fx):
    return (y - fx) ** 2


def log_loss(y, fx):
    return log(1 + exp(-y * fx))


def boosting_optimized_rule_ensemble(n, d, k, L, U, x, y, labels, reg=0, loss_func='squared', tl=100, f=None):
    xx = pd.DataFrame(x, columns=labels)
    yy = pd.Series(y)
    rbe = RuleBoostingEstimator(num_rules=k,
                                base_learner=XGBRuleEstimator(loss=loss_func, reg=reg,
                                                              search_params={'max_col_attr': 10}))
    rules = rbe.fit(xx, yy).rules_
    fx = rules(xx)
    loss = squared_loss if loss_func == 'squared' else log_loss
    boosting_risk = reg / 2 / n * sum([r.y * r.y for r in rules.members]) + sum(
        [loss(yy[i], fx[i]) for i in range(n)]) / n
    if f is not None:
        f.write('=====boosting====\n')
        f.write(repr(rules) + "\n")
        f.write('boosting risk: ' + str(boosting_risk) + '\n')
        print('boosting risk: ' + str(boosting_risk) + '\n')
    print(rules)
    init_w = []
    init_u = [[x for x in U] for _ in range(k)]
    init_l = [[x for x in L] for _ in range(k)]

    # U = [x + 0.001 if type(x) == float else x for x in U]
    # L = [x - 0.001 if type(x) == float else x for x in L]
    for i in range(len(rules.members)):
        r = rules.members[i]
        # print(r.y)
        init_w.append(float(r.y))
        for qs in r.q.props:
            # print(qs.repr)
            if '<=' in qs.repr:
                parts = qs.repr.split('<=')
                # print(parts)
                init_u[i][labels.index(parts[0])] = float(parts[1])
            elif '>=' in qs.repr:
                parts = qs.repr.split('>=')
                # print(parts)
                init_l[i][labels.index(parts[0])] = float(parts[1])
    print(init_w, init_l, init_u)
    print(L, U)
    if f is not None:
        f.write("======optimization======\n")
    return optimized_rule_ensemble2(n, d, k, L, U, x, y, labels, reg=reg, loss_func=loss_func, tl=tl,
                                    init_w=init_w, init_l=init_l, init_u=init_u, f=f)[0], rules, boosting_risk


if __name__ == '__main__':
    labels = ['GDP']
    seeds = get_splits()['gdp']
    train, test, train_target, test_target, L, U, d, n = preprocess_pd(
        '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
        labels, [int], 'Satisfaction', target_type=float, random_seed=seeds[0])
    print(train)
    print(train_target)
    weights = [-1.5008928568227151, 7.143750000378351, 0.6171428598902571]
    lowers = [[9437], [9009], [12495]]
    uppers = [[32486], [101994], [17257]]
    l, u, w, yy, obj, bnd = optimized_rule_ensemble2(n, d, 3, L, U, train, train_target, ['GDP'], tl=500,
                                                     init_w=weights,
                                                     init_u=uppers, init_l=lowers)
    sum1 = 0
    sum2 = 0
    print(w)
    print(l)
    print(u)
    print(yy)
    ensemble = build_ensemble(w, l, u, L, U, ['GDP'])
    train_df = pd.DataFrame(train, columns=labels)
    yyy = ensemble(train_df)
    print(yyy)
    for i in range(len(train)):
        sum1 += (yy[i] - train_target[i]) ** 2
        sum2 += (yyy[i] - train_target[i]) ** 2
    print(sum1 / len(yy), sum2 / len(yyy))
