import copy
import sys
from datetime import datetime
from math import log

import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from realkd.rules import AdditiveRuleEnsemble, loss_function

from evaluation.data_info import get_splits
from boosting_col2 import boosting_step2
from build_rule_ensemble import build_ensemble
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2

from evaluation.data_preprocess import preprocess_pd


def fully_corrective2(n, d, k, L, U, x, y, labels, init_w, init_l, init_u, reg=0.0, loss_func='squared', tl=100, f=None,
                      max_col_num=10):
    m = gp.Model("rule1")
    log_loss_y = [4.0181499279178094, 3.048587351573742, 2.1269280110429727, 1.3132616875182228,
                  0.6931471805599453, 0.31326168751822286, 0.1269280110429726, 0.04858735157374196,
                  0.01814992791780978, 0, 0]
    if loss_func != 'squared':
        m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.TimeLimit, tl)
    weight = m.addVars(k, vtype=GRB.CONTINUOUS, lb=-max([max(y),-min(y)])*5, name="weight")
    s = m.addVars(n, k, d, vtype=GRB.BINARY, name="s")
    z = m.addVars(n, k, vtype=GRB.BINARY, name="z")
    w = m.addVars(n, k, vtype=GRB.CONTINUOUS, lb=-max([max(y),-min(y)])*5, name="w")
    yy = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-max([max(y),-min(y)])*5, name="yy")
    loss = m.addVars(n, vtype=GRB.CONTINUOUS, name="loss")
    mult = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-max([max(y),-min(y)])*5, name="mult")
    obj = m.addVar(name='obj', vtype=GRB.CONTINUOUS)
    t = m.addVars(n, k, d, vtype=GRB.BINARY, name='t')
    u = m.addVars(n, k, d, vtype=GRB.BINARY, name='u')
    m.update()
    epsilon = 0.0001
    round_value = -round(log(epsilon, 10))
    for r in range(k):
        weight[r].start = init_w[r]
    for i in range(n):
        for r in range(len(init_l)):
            for j in range(d):
                if init_l[r][j] <= x[i][j] <= init_u[r][j]:
                    m.addConstr(s[i, r, j] == 1, name='init_s' + str(i) + str(r) + str(j))
                else:
                    m.addConstr(s[i, r, j] == 0, name='init_s' + str(i) + str(r) + str(j))
    for r in range(k):
        for j in range(d):
            for i in range(n):
                m.addConstr(z[i, r] <= s[i, r, j], name='z1-' + str(r) + str(j) + str(i))
        for i in range(n):
            m.addConstr(z[i, r] >= gp.quicksum(s[i, r, j] for j in range(d)) - d + 1, name='z2-' + str(r) + str(i))
            m.addConstr(w[i, r] == weight[r] * z[i, r], name='w' + str(r) + str(i))
    for i in range(n):
        # m.addConstr(gp.quicksum(z[i,r] for r in range(k))<=2)
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
    res_weights = [weight[wi].x for wi in range(k)]
    for r in range(k):  # rule
        res_l = []
        res_u = []
        print('%g if ' % weight[r].x, end='')
        if f is not None:
            f.write('%g if ' % weight[r].x)
        for j in range(d):  # dim
            if init_l[r][j] != L[j] or init_u[r][j] != U[j]:
                if init_l[r][j] != init_u[r][j]:
                    print('%g <= %s <= %g ' % (init_l[r][j], labels[j], init_u[r][j]), end='')
                    if f is not None:
                        f.write('%g <= %s <= %g ' % (init_l[r][j], labels[j], init_u[r][j]))
                else:
                    print('%s = %g ' % (labels[j], init_u[r][j]), end='')
                    if f is not None:
                        f.write('%s = %g ' % (labels[j], init_u[r][j]))
            res_l.append(round(init_l[r][j], round_value))
            res_u.append(round(init_u[r][j], round_value))
        print('')
        if f is not None:
            f.write('\n')
    return res_weights, [yy[yi].x for yi in yy], m.getObjective().getValue(), m.ObjBound


def fully_corrective_boosting2(n, d, k, L, U, x, y, labels, reg=0.0, loss_func='squared', tl=100, f=None, left_most=0,
                               debug=False, max_col_num=10):
    ensemble = AdditiveRuleEnsemble([])
    ensembles = []
    weights = []
    lowers = []
    uppers = []
    risk = 1e10
    for i in range(1, k + 1):
        print('=====iteration ' + str(i) + '========')
        print('======boosting==========')
        if f is not None:
            f.write('=====iteration ' + str(i) + '========\n')
            f.write('======boosting==========\n')
        weights, lowers, uppers, yy, risk, bnd = boosting_step2(n, d, i, L, U, x, y, labels, reg=reg,
                                                                loss_func=loss_func,
                                                                tl=tl,
                                                                f=f, init_w=weights, init_l=lowers, init_u=uppers,
                                                                left_most=left_most, debug=debug,
                                                                max_col_num=max_col_num)
        # print('actual risk boosting: ', sum((yy[j] - y[j]) ** 2 for j in range(n)) / n)
        if f is not None:
            f.write('boosting risk = ' + str(risk) + '\n')
            f.write('boosting result\n')
            f.write('weights: ' + str(weights) + '\n')
            f.write('lowers: ' + str(lowers) + '\n')
            f.write('uppers: ' + str(uppers) + '\n')
            f.write('ensemble: ' + str(build_ensemble(weights, lowers, uppers, L, U, labels)) + '\n')
            f.write('bound: ' + str(bnd) + '\n')
            print(str(build_ensemble(weights, lowers, uppers, L, U, labels)))
            f.write('==========fully corrective=========\n')
        print('==========fully corrective=========')
        weights, yyy, risk_fc, bnd = fully_corrective2(n, d, i, L, U, x, y, labels, weights, lowers, uppers, reg=reg,
                                                       loss_func=loss_func, tl=tl, f=f, max_col_num=max_col_num)
        ensemble = build_ensemble(weights, lowers, uppers, L, U, labels)
        ensembles.append(ensemble)
        if f is not None:
            f.write('fc boosting risk = ' + str(risk_fc) + '\n')
            f.write('fc boosting result\n')
            f.write('weights: ' + str(weights) + '\n')
            f.write('lowers: ' + str(lowers) + '\n')
            f.write('uppers: ' + str(uppers) + '\n')
            f.write('boundary: ' + str(bnd) + '\n')
            f.write('ensemble: ' + str(ensemble) + '\n\n\n')
    return ensemble, risk, ensembles, bnd


def evaluate_fc_boosting2(dataset_name, path, labels, feature_types, target, target_type, cr='r', feature_map={},
                          loss='squared', tl=100, repeat=5, max_rule_num=5, debug=False, reg=0.0, max_col_num=10):
    seeds = get_splits()[dataset_name]
    k = max_rule_num
    func = skl_auc if cr == 'c' else skl_r2
    for m in range(repeat):
        file = open(
            "../output20240924fc2/" + dataset_name  + "_fc_boosting2_no_priority_sym_ind_950_reg" + str(
                reg) + "tl" + str(tl) + 'col' + str(max_col_num)+'rep'+str(m) + ".txt", "w")
        # original_stdout = sys.stdout
        with open("../output20240924fc2/" + dataset_name + "_output_fc_boosting2_no_priority_sym_ind_950_reg" + str(reg) + "tl" + str(
            tl) + 'col' + str(max_col_num) +'rep'+str(m)+ ".txt", "w") as f:
            # sys.stdout = f
            risks = []
        
            file.write("======Dataset " + str(m) + "=======\n")
            train, test, train_target, test_target, L, U, d, n = preprocess_pd(path,
                                                                               labels,
                                                                               feature_types,
                                                                               target, target_type=target_type,
                                                                               feature_map=feature_map,
                                                                               random_seed=seeds[m])
            ensemble, risk, ensembles, bnd = fully_corrective_boosting2(n, d, k, L, U, train, train_target, labels,
                                                                        loss_func=loss, tl=tl, reg=reg,
                                                                        f=file, debug=debug, max_col_num=max_col_num)
            file.write("num     test        train\n")
            print("num     test        train\n============================")
            i = 0
            loss_func = loss_function(loss)
            for rule_ensemble in ensembles:
                train_df = pd.DataFrame(train, columns=labels)
                test_df = pd.DataFrame(test, columns=labels)
                train_sr = pd.Series(train_target)
                test_sr = pd.Series(test_target)
                # test_score = func(test_target, rule_ensemble(test_df))
                # train_score = func(train_target, rule_ensemble(train_df))
                test_score = sum(loss_func(rule_ensemble(test_df), test_sr)) / len(test_sr)
                train_score = sum(loss_func(rule_ensemble(train_df), train_sr)) / n
                file.write(str(i) + " " + str(test_score) + " " + str(train_score) + "\n")
                print(str(i) + " " + str(test_score) + " " + str(train_score))
                i += 1
            file.write(repr(ensemble) + "\n")

            risks.append(risk)
        file.close()
        print(dataset_name, 'fc, model2, fixed, no priority, symmetry, indicator, 950, tl=', tl, 'reg=', reg)
        # sys.stdout = original_stdout
    print(dataset_name, risks)
    return risks

