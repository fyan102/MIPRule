import copy
from datetime import datetime
from math import log, exp
import sys
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from realkd.rules import AdditiveRuleEnsemble, loss_function
from sklearn.datasets import load_wine, load_iris, load_diabetes, load_breast_cancer

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd, preprocess_datasets
from build_rule_ensemble import build_ensemble
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2


def boosting_step2(n, d, k, L, U, x, y, labels, reg=0.0, loss_func='squared', tl=100,
                   init_w=[], init_l=[], init_u=[], f=None, epsilon=0.00001, left_most=0, debug=False, max_col_num=10):
    new_x = []
    for i in range(len(x)):
        new_x.append([i] + x[i].tolist())
    m = gp.Model("rule1")
    log_loss_x = [x / 8 for x in range(-40, 40, 1)]
    log_loss_y = [log((1 + exp(-x)), 2) for x in log_loss_x] + [0, 0]
    log_loss_x += [5, 6]
    round_value = -round(log(epsilon, 10))
    m.setParam(GRB.Param.TimeLimit, tl)
    weight = m.addVars(k, vtype=GRB.CONTINUOUS, lb=-max([max(y), -min(y)]) * 5, name="weight")
    s = m.addVars(n, k, d, vtype=GRB.BINARY, name="s")
    z = m.addVars(n, k, vtype=GRB.BINARY, name="z")
    w = m.addVars(n, k, vtype=GRB.CONTINUOUS, lb=-max([max(y), -min(y)]) * 5, name="w")
    yy = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-max([max(y), -min(y)]) * 5, name="yy")
    loss = m.addVars(n, vtype=GRB.CONTINUOUS, name="loss")
    mult = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-max([max(y), -min(y)]) * 5, name="mult")
    obj = m.addVar(name='obj', vtype=GRB.CONTINUOUS)
    t = m.addVars(n, k, d, vtype=GRB.BINARY, name='t')
    u = m.addVars(n, k, d, vtype=GRB.BINARY, name='u')
    m.update()

    # boosting init from last rule ensemble
    if len(init_w) != 0:
        for r in range(len(init_w)):
            m.addConstr(weight[r] == init_w[r], name='init_w' + str(r))
    if len(init_l) != 0 and len(init_u) != 0:
        for i in range(n):
            for r in range(len(init_l)):
                for j in range(d):
                    if init_l[r][j] <= x[i][j] <= init_u[r][j]:
                        m.addConstr(s[i, r, j] == 1,
                                    name='init_s' + str(i) + str(r) + str(j))
                    else:
                        m.addConstr(s[i, r, j] == 0,
                                    name='init_s' + str(i) + str(r) + str(j))
                    if init_l[r][j] <= x[i][j]:
                        m.addConstr(t[i, r, j] == 1)
                    else:
                        m.addConstr(t[i, r, j] == 0)
                    if init_u[r][j] >= x[i][j]:
                        m.addConstr(u[i, r, j] == 1)
                    else:
                        m.addConstr(u[i, r, j] == 0)
    # constraints for u, t, s
    for j in range(d):
        temp_x = sorted(new_x, key=lambda row: row[j + 1])
        # interval_len = (U[j] - L[j]) / max_col_num
        vals = [nx[j + 1] for nx in new_x]
        unique_values = np.unique(vals)
        if len(unique_values) * 2 > max_col_num:
            intervals = sorted(pd.qcut(vals, max_col_num // 2, retbins=True, duplicates='drop')[1:])[0]
        else:
            intervals = unique_values
        print('Intervals:', intervals)
        for r in range(k - 1, k):
            m.addConstr(gp.quicksum(s[i, r, j] for i in range(n)) >= 1, name='sum_s_ge_1_' + str(j) + str(r))
            # for ii in range(max_col_num):
            for ii in range(len(intervals) - 1):
                for i in range(n):
                    # if i + 1 < n and ii * interval_len + L[j] <= temp_x[i][j + 1] < (
                    #         ii + 1) * interval_len + L[j] and ii * interval_len + L[j] <= \
                    #         temp_x[i + 1][j + 1] < (ii + 1) * interval_len + L[j]:
                    #     m.addConstr(t[temp_x[i][0], r, j] == t[temp_x[i + 1][0], r, j])
                    #     m.addConstr(u[temp_x[i][0], r, j] == u[temp_x[i + 1][0], r, j])
                    # if temp_x[i][j + 1] >= (ii + 1) * interval_len + L[j]:
                    #     break
                    if i + 1 < n and intervals[ii] <= temp_x[i][j + 1] < intervals[ii + 1] and \
                            intervals[ii] <= temp_x[i + 1][j + 1] < intervals[ii + 1]:
                        m.addConstr(t[temp_x[i][0], r, j] == t[temp_x[i + 1][0], r, j])
                        m.addConstr(u[temp_x[i][0], r, j] == u[temp_x[i + 1][0], r, j])
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
                m.addConstr(s[i, r, j] >= t[i, r, j] + u[i, r, j] - 1, name='s1-' + str(r) + str(j) + str(i))
                m.addConstr(s[i, r, j] <= t[i, r, j], name='s2-' + str(r) + str(j) + str(i))
                m.addConstr(s[i, r, j] <= u[i, r, j], name='s3-' + str(r) + str(j) + str(i))
                m.addConstr(z[i, r] <= s[i, r, j], name='z1-' + str(r) + str(j) + str(i))
        # constraint for z 1
        for r in range(k):
            m.addConstr(gp.quicksum(s[i, r, j] for i in range(n)) >= 1, name='sum_s_ge_1_' + str(j) + str(r))
            for i in range(n):
                m.addConstr(z[i, r] <= s[i, r, j], name='z1-' +
                                                        str(r) + str(j) + str(i))
    # constraint for z and w (indicator)
    for r in range(k):
        for i in range(n):
            m.addConstr(z[i, r] >= gp.quicksum(s[i, r, j]
                                               for j in range(d)) - d + 1, name='z2-' + str(r) + str(i))
            # m.addConstr((z[i, r] == 1) >> (w[i, r] == weight[r]),
            #             name='w' + str(r) + str(i))
            # m.addConstr((z[i, r] == 0) >> (w[i, r] == 0))
            m.addConstr(w[i, r] == weight[r] * z[i, r], name='w' + str(r) + str(i))
    # constraint for predicted y, objective functions
    for i in range(n):
        m.addConstr(yy[i] == gp.quicksum(w[i, r]
                                         for r in range(k)), name='yy' + str(i))
        if loss_func == 'logistic':
            m.addConstr(mult[i] == yy[i] * y[i], name='multi' + str(i))
            m.addGenConstrPWL(
                mult[i], loss[i], log_loss_x, log_loss_y, 'log_loss')
    if loss_func == "squared":
        m.setObjective((gp.quicksum((yy[i] - y[i]) * (yy[i] - y[i]) for i in range(n)) + reg * gp.quicksum(
            weight[i] * weight[i] for i in range(k)) / 2) / n)
    elif loss_func == 'logistic':
        m.setObjective((gp.quicksum(loss[i] for i in range(n)) + reg * gp.quicksum(
            weight[i] * weight[i] for i in range(k)) / 2) / n)
    m.optimize()
    # print the values of s, t, v, debug
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
    # get the lower and upper bounds
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
    # weights
    res_weights = [weight[wi].x for wi in range(k)]
    # print rules
    for r in range(k):  # rule
        res_l = []
        res_u = []
        print('%g if ' % (weight[r].x), end='')
        if f is not None and r == k - 1:
            f.write('%g if ' % weight[r].x)
        for j in range(d):  # dim
            if res_lowers[r][j] != L[j] or res_uppers[r][j] != U[j]:
                if res_lowers[r][j] != res_uppers[r][j]:
                    print('%g <= %s <= %g ' %
                          (res_lowers[r][j], labels[j], res_uppers[r][j]), end='')
                    if f is not None and r == k - 1:
                        f.write('%g <= %s <= %g ' %
                                (res_lowers[r][j], labels[j], res_uppers[r][j]))
                else:
                    print('%s = %g ' % (labels[j], res_uppers[r][j]), end='')
                    if f is not None and r == k - 1:
                        f.write('%s = %g ' % (labels[j], res_uppers[r][j]))
            res_l.append(round(res_lowers[r][j], round_value))
            res_u.append(round(res_uppers[r][j], round_value))
        print('')
    # risk and bounds
    if f is not None:
        f.write('\n')
        f.write('opt boosting risk: ' + str(m.getObjective().getValue()) + "\n")
        f.write('opt bound: ' + str(m.ObjBound) + '\n')
    return res_weights, res_lowers, res_uppers, [yy[yi].X for yi in yy], m.getObjective().getValue(), m.ObjBound


def rule_boosting2(n, d, k, L, U, x, y, labels, reg=0.0, loss_func='squared', tl=100, f=None, left_most=0, debug=False,
                   max_col_num=10):
    ensemble = AdditiveRuleEnsemble([])
    w = []
    l = []
    u = []
    risk = 1e10
    bnd = 0
    # iterations
    for i in range(1, 1 + k):
        w, l, u, yy, risk, bnd = boosting_step2(n, d, i, L, U, x, y, labels, reg=reg, loss_func=loss_func, tl=tl,
                                                init_w=w,
                                                init_l=l, init_u=u, f=f, left_most=left_most, debug=debug,
                                                max_col_num=max_col_num)
        print(w, l, u)
        ensemble = build_ensemble(w, l, u, L, U, labels)
        # y_diff = [y_diff[j] - yy[j] for j in range(len(y_diff))]
    return ensemble, risk, bnd


def evaluate_boosting2(dataset_name, load_method, cr='r', feature_map={},
                       loss='squared', tl=500, repeat=5, max_rule_num=5, reg=0.0, debug=False, max_col_num=10):
    seeds = get_splits()[dataset_name]
    k = max_rule_num
    func = skl_auc if cr == 'c' else skl_r2
    file = open("../output20250524bc2/" + dataset_name +
                str(datetime.now()) + "_boosting2_no_priority_sym_ind_950_reg" + str(reg) + "tl" + str(
        tl) + 'col' + str(max_col_num) + ".txt",
                "w")
    original_stdout = sys.stdout
    with open("../output20250524bc2/" + dataset_name +
              str(datetime.now()) + "_output_boosting2_no_priority_sym_ind_950_reg" + str(reg) + "tl" + str(
        tl) + 'col' + str(max_col_num) + ".txt", "w") as f:
        # sys.stdout = f
        for m in range(repeat):
            train, test, train_target, test_target, L, U, d, n, labels = preprocess_datasets(load_method,
                                                                                             feature_map=feature_map,
                                                                                             random_seed=seeds[m])
            ensemble, risk, bnd = rule_boosting2(
                n, d, k, L, U, train, train_target, labels, loss_func=loss, tl=tl, f=file, reg=reg, debug=debug,
                max_col_num=max_col_num)
            file.write("num    test        train\n")
            loss_func = loss_function(loss)
            for i in range(1, 1 + k):
                rule_ensemble = AdditiveRuleEnsemble(ensemble.members[:i + 1])
                train_df = pd.DataFrame(train, columns=labels)
                test_df = pd.DataFrame(test, columns=labels)
                train_sr = pd.Series(train_target)
                test_sr = pd.Series(test_target)
                # test_score = func(test_target, rule_ensemble(test_df))
                # train_score = func(train_target, rule_ensemble(train_df))
                test_score = sum(loss_func(rule_ensemble(test_df), test_sr)) / len(test_sr)
                train_score = sum(loss_func(rule_ensemble(train_df), train_sr)) / n

                file.write(str(i) + " " + str(test_score) +
                           " " + str(train_score) + "\n")
            file.write(repr(ensemble) + "\n")
            file.write(dataset_name + " booosting, model 2, no z, indicator, 950")
        file.close()
        print(dataset_name, 'boosting, model2, fixed, no priority, symmetry, indicator, 950, tl=', tl, 'reg=', reg)
        sys.stdout = original_stdout
    print(dataset_name, risk)
    return risk


if __name__ == '__main__':
    res = {}
    for col in [10]:  # finished
            res['load_wine'] = evaluate_boosting2('load_wine', load_wine, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                        repeat=1, reg=1, loss='logistic',
                                        max_rule_num=10, max_col_num=col)

    for col in [10]:  # finished
        try:
            res['iris'] = evaluate_boosting2('iris', load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                   repeat=1, reg=1, loss='logistic',
                                   max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [5]:  # finished
        try:
            res['diabetes'] = evaluate_boosting2('load_diabetes', load_diabetes,
                                       repeat=1,
                                       max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [5]:
        try:
            res['breast'] = evaluate_boosting2('breast_cancer', load_breast_cancer,
                                     repeat=1, reg=1, loss='logistic',
                                     max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
