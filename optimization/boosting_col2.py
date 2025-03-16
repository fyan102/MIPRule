import copy
from datetime import datetime
from math import log, exp
import sys
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from realkd.rules import AdditiveRuleEnsemble, loss_function

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd
from build_rule_ensemble import build_ensemble
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2


def boosting_step2(n, d, k, L, U, x, y, labels, reg=0.0, loss_func='squared', tl=100,
                   init_w=[], init_l=[], init_u=[], f=None, epsilon=0.00001, left_most=0, debug=False, max_col_num=10):
    new_x = []
    for i in range(len(x)):
        new_x.append([i] + x[i])
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


def evaluate_boosting2(dataset_name, path, labels, feature_types, target, target_type, cr='r', feature_map={},
                       loss='squared', tl=100, repeat=5, max_rule_num=5, reg=0.0, debug=False, max_col_num=10):
    seeds = get_splits()[dataset_name]
    k = max_rule_num
    func = skl_auc if cr == 'c' else skl_r2

    # sys.stdout = f
    for m in range(repeat):
        file = open("../output20241024bc2/" + dataset_name +
                    "_boosting2_no_priority_sym_ind_reg" + str(reg) + "tl" + str(
            tl) + 'col' + str(max_col_num) + 'rep' + str(m) + ".txt",
                    "w")
        original_stdout = sys.stdout
        with open("../output20241024bc2/" + dataset_name +
                  "_output_boosting2_no_priority_sym_ind_reg" + str(reg) + "tl" + str(
            tl) + 'col' + str(max_col_num) + 'rep' + str(m) + ".txt", "w") as f:
            train, test, train_target, test_target, L, U, d, n = preprocess_pd(path,
                                                                               labels,
                                                                               feature_types,
                                                                               target, target_type=target_type,
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
        # sys.stdout = original_stdout
    print(dataset_name, risk)
    return risk


if __name__ == '__main__':
    res = {}
    for m in [10]:
        for r in [0]:
            res['gdp'] = evaluate_boosting2('gdp', '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
                                            ['GDP'], [int], 'Satisfaction', target_type=float, tl=250, repeat=5,
                                            max_rule_num=10, debug=False, reg=r, max_col_num=m)
        for r in [1]:
            res['wage'] = evaluate_boosting2('wage', '../datasets/wages_demographics/wages.csv',
                                             ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int],
                                             'earn',
                                             target_type=float,
                                             feature_map={'sex': {'male': 1, 'female': 0},
                                                          'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}},
                                             tl=250, repeat=5, max_rule_num=10, reg=r, max_col_num=m)
        for r in [0.2]:
            res['titanic'] = evaluate_boosting2('titanic', '../datasets/titanic/train.csv',
                                                ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                                                [int, str, float, int, int, float, str], 'Survived', target_type=int,
                                                cr='c',
                                                feature_map={'Sex': {'male': 1, 'female': 0},
                                                             'Embarked': {'S': 1, 'C': 2, 'Q': 3},
                                                             'Survived': {'0': -1, '1': 1}}, loss='logistic', tl=250,
                                                repeat=5,
                                                max_rule_num=10, reg=r, max_col_num=m)
        for r in [0]:
            res['insurance'] = evaluate_boosting2('insurance', '../datasets/insurance/insurance.csv',
                                                  ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                                                  [int, str, float, int, str, str], 'charges', target_type=float,
                                                  feature_map={'sex': {'male': 1, 'female': 0},
                                                               'smoker': {'yes': 1, 'no': 0},
                                                               'region': {'southwest': 1, 'southeast': 2,
                                                                          'northwest': 3,
                                                                          'northeast': 4}},
                                                  tl=250, repeat=5, max_rule_num=10, reg=r, max_col_num=m)
        for r in [0]:
            res['used_cars'] = evaluate_boosting2('used_cars',
                                                  '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
                                                  ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
                                                  target_type=float,
                                                  tl=250, repeat=5, max_rule_num=10, reg=r, debug=False)
        for r in [0.1]:
            res['tic_tac_toe'] = evaluate_boosting2('tic-tac-toe', '../datasets/tic_tac_toe/tic_tac_toe.csv',
                                                    ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
                                                    [str, str, str, str, str, str, str, str, str], 'V10',
                                                    target_type=str,
                                                    cr='c',
                                                    feature_map={'V1': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V2': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V3': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V4': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V5': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V6': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V7': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V8': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V9': {'x': 1, 'o': 2, 'b': 3},
                                                                 'V10': {'positive': 1, 'negative': -1}},
                                                    loss='logistic', tl=250, repeat=5, max_rule_num=10, reg=r)
        for r in [0]:
            res['boston'] = evaluate_boosting2('boston', '../datasets/boston/boston_house_prices.csv',
                                               ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                                'PTRATIO',
                                                'B',
                                                'LSTAT'],
                                               [float, float, float, float, float, float, float, int, int, float, float,
                                                float],
                                               'MEDV', max_col_num=4,
                                               target_type=float, tl=250, repeat=5, max_rule_num=10, reg=r)
        print(res)
    for col in [10]:  # finished
        try:
            res['world_happiness_indicator'] = evaluate_boosting2('world_happiness_indicator',
                                                                  '../datasets/world_happiness_indicator/2019.csv',
                                                                  ['GDP per capita', 'Social support',
                                                                   'Healthy life expectancy',
                                                                   'Freedom to make life choices',
                                                                   'Generosity', 'Perceptions of corruption'],
                                                                  [float, float, float, float, float, float, ], 'Score',
                                                                  target_type=float, tl=250, reg=0,
                                                                  repeat=5, max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [4]:
        try:
            res['Demographics'] = evaluate_boosting2('Demographics', '../datasets/Demographics/Demographics1.csv',
                                                     ['Sex', 'Marital', 'Age', 'Edu', 'Occupation', 'LivingYears',
                                                      'Persons',
                                                      'PersonsUnder18', 'HouseholderStatus',
                                                      'TypeOfHome', 'Ethnic', 'Language'],
                                                     [str, str, int, int, str, int, int, int, str, str, str, str],
                                                     'AnnualIncome',
                                                     target_type=int,
                                                     feature_map={'Sex': {' Male': 1, ' Female': 0},
                                                                  'Marital': {' Married': 1, '': 0,
                                                                              ' Single, never married': 2,
                                                                              ' Divorced or separated': 3,
                                                                              ' Living together, not married': 4,
                                                                              ' Widowed': 5},
                                                                  'Occupation': {'': 0, ' Homemaker': 1,
                                                                                 ' Professional/Managerial': 2,
                                                                                 ' Student, HS or College': 3,
                                                                                 ' Retired': 4, ' Unemployed': 5,
                                                                                 ' Factory Worker/Laborer/Driver': 6,
                                                                                 ' Sales Worker': 7,
                                                                                 ' Clerical/Service Worker': 8,
                                                                                 ' Military': 9},
                                                                  'HouseholderStatus': {'': 0, ' Own': 1, ' Rent': 2,
                                                                                        ' Live with Parents/Family': 3},
                                                                  'TypeOfHome': {'': 0, ' House': 1,
                                                                                 ' Apartment': 2,
                                                                                 ' Condominium': 3,
                                                                                 ' Mobile Home': 4, ' Other': 5, },
                                                                  'Ethnic': {'': 0, ' White': 1,
                                                                             ' Hispanic': 2,
                                                                             ' Asian': 3,
                                                                             ' Black': 4, ' East Indian': 5,
                                                                             ' Pacific Islander': 6,
                                                                             ' American Indian': 7,
                                                                             ' Other': 8, },
                                                                  'Language': {'': 0, ' English': 1, ' Spanish': 2,
                                                                               ' Other': 3, }
                                                                  }, tl=250, reg=0,
                                                     repeat=5, max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)

    for col in [3]:
        res['IBM_HR'] = evaluate_boosting2('IBM_HR', '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                           ["Age", 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                                            'Education',
                                            'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                                            'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                                            'MonthlyIncome',
                                            'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                                            'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
                                            'StockOptionLevel',
                                            'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                                            'YearsAtCompany',
                                            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
                                           [int, str, int, str, int, int, str, int, str, int, int, int, str, int, str,
                                            int,
                                            int,
                                            int, str, int, int, int, int, int, int, int, int, int, int, int, int],
                                           'Attrition', target_type=str,
                                           feature_map={"BusinessTravel": {'Travel_Rarely': 1, 'Travel_Frequently': 2,
                                                                           'Non-Travel': 3},
                                                        "Attrition": {'Yes': 1, 'No': 0},
                                                        'Department': {'Sales': 1, 'Research & Development': 2,
                                                                       'Human Resources': 3},
                                                        'EducationField': {'Life Sciences': 1, 'Medical': 2,
                                                                           'Marketing': 3,
                                                                           'Technical Degree': 4, 'Human Resources': 5,
                                                                           'Other': 6},
                                                        'Gender': {'Male': 1, 'Female': 0},
                                                        'JobRole': {'Sales Executive': 1, 'Research Scientist': 2,
                                                                    'Laboratory Technician': 3,
                                                                    'Manufacturing Director': 4,
                                                                    'Healthcare Representative': 5,
                                                                    'Manager': 6, 'Human Resources': 7,
                                                                    'Research Director': 8,
                                                                    'Sales Representative': 9},
                                                        'MaritalStatus': {'Single': 1, 'Married': 2, 'Divorced': 3},
                                                        'OverTime': {'Yes': 1, 'No': -1}, },
                                           loss='logistic', tl=250, reg=0.05, cr='c',
                                           repeat=5, max_rule_num=10, max_col_num=col)

    for col in [4]:
        try:
            res['telco_churn'] = evaluate_boosting2('telco_churn',
                                                    '../datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                                                    ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                                     'PhoneService', 'MultipleLines', 'InternetService',
                                                     'OnlineSecurity',
                                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                                     'MonthlyCharges', 'TotalCharges', ],
                                                    [str, int, str, str, int, str, str, str, str, str, str, str, str,
                                                     str,
                                                     str, str, str, float, float],
                                                    'Churn',
                                                    target_type=str,
                                                    feature_map={'gender': {'Male': 1, 'Female': 0},
                                                                 'Partner': {'Yes': 1, 'No': 0},
                                                                 'Dependents': {'Yes': 1, 'No': 0},
                                                                 'PhoneService': {'Yes': 1, 'No': 0},
                                                                 'MultipleLines': {'Yes': 1, 'No': 2,
                                                                                   'No phone service': 3},
                                                                 'InternetService': {'DSL': 1, 'Fiber optic': 2,
                                                                                     'No': 3},
                                                                 'OnlineSecurity': {'Yes': 1, 'No': 2,
                                                                                    'No internet service': 3},
                                                                 'OnlineBackup': {'Yes': 1, 'No': 2,
                                                                                  'No internet service': 3},
                                                                 'DeviceProtection': {'Yes': 1, 'No': 2,
                                                                                      'No internet service': 3},
                                                                 'TechSupport': {'Yes': 1, 'No': 2,
                                                                                 'No internet service': 3},
                                                                 'StreamingTV': {'Yes': 1, 'No': 2,
                                                                                 'No internet service': 3},
                                                                 'StreamingMovies': {'Yes': 1, 'No': 2,
                                                                                     'No internet service': 3},
                                                                 'Contract': {'Month-to-month': 1, 'One year': 2,
                                                                              'Two year': 3, },
                                                                 'PaperlessBilling': {'Yes': 1, 'No': 0},
                                                                 'PaymentMethod': {'Electronic check': 1,
                                                                                   'Mailed check': 2,
                                                                                   'Bank transfer (automatic)': 3,
                                                                                   'Credit card (automatic)': 4},
                                                                 'Churn': {'Yes': 1, 'No': -1},
                                                                 }, tl=250, reg=0.05, cr='c', loss='logistic',
                                                    repeat=5, max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)

    for col in [4]:
        try:
            res['mobile_prices'] = evaluate_boosting2('mobile_prices',
                                                      '../datasets/mobile_prices/train.csv',
                                                      ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                                                       'four_g',
                                                       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                                                       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                                                       'touch_screen',
                                                       'wifi'],
                                                      [int, int, float, int, int, int, int, float, int, int, int, int,
                                                       int, int,
                                                       int,
                                                       int, int, int, int, int, ], 'price_range',
                                                      target_type=int,
                                                      tl=250, reg=0,
                                                      repeat=5, max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [4]:
        try:
            res['GenderRecognition'] = evaluate_boosting2('GenderRecognition',
                                                          '../datasets/GenderRecognition/voice.csv',
                                                          ["meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew",
                                                           "kurt",
                                                           "sp.ent", "sfm", "mode", "centroid", "meanfun", "minfun",
                                                           "maxfun",
                                                           "meandom", "mindom", "maxdom", "dfrange", "modindx"],
                                                          [float, float, float, float, float, float, float, float,
                                                           float,
                                                           float,
                                                           float, float, float, float, float, float, float, float,
                                                           float,
                                                           float],
                                                          "label", target_type=str,
                                                          feature_map={"label": {'male': 1, 'female': -1}}, tl=250,
                                                          reg=0.1,
                                                          cr='c', loss='logistic',
                                                          repeat=5, max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for m in [10]:
        for r in [0]:
            res['social_media'] = evaluate_boosting2('social_media', '../datasets/social_media/social-media.csv',
                                                     ['UsageDuraiton', 'Age'], [int, int], 'TotalLikes',
                                                     target_type=float, tl=250, repeat=5,
                                                     max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['salary'] = evaluate_boosting2('salary', '../datasets/salary/Salary_dataset.csv',
                                               ['YearsExperience'], [float], 'Salary',
                                               target_type=float, tl=250, repeat=5,
                                               max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['student_marks'] = evaluate_boosting2('student_marks', '../datasets/student_marks/Student_Marks.csv',
                                                      ['number_courses', 'time_study'], [int, float], 'Marks',
                                                      target_type=float, tl=250, repeat=5,
                                                      max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['study_time'] = evaluate_boosting2('study_time', '../datasets/study_time/score_updated.csv',
                                                   ['Hours'], [float], 'Scores',
                                                   target_type=float, tl=250, repeat=5,
                                                   max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['income'] = evaluate_boosting2('income', '../datasets/income/multiple_linear_regression_dataset.csv',
                                               ['age', 'experience'], [int, int], 'income',
                                               target_type=float, tl=250, repeat=5,
                                               max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['headbrain'] = evaluate_boosting2('headbrain', '../datasets/headbrain/headbrain.csv',
                                                  ['Gender', 'Age Range', 'Head Size(cm^3)'], [int, int, int],
                                                  'Brain Weight(grams)',
                                                  target_type=float, tl=250, repeat=5,
                                                  max_rule_num=10, debug=False, reg=r, max_col_num=m)
            res['fitness'] = evaluate_boosting2('fitness', '../datasets/fitness/data.csv',
                                                ['Duration', 'Pulse', 'Maxpulse'], [int, int, int], 'Calories',
                                                target_type=float, tl=250, repeat=5,
                                                max_rule_num=10, debug=False, reg=r, max_col_num=m)
            # res[''] = evaluate_boosting2('', '../datasets/',
            #                              [''], [], '',
            #                              target_type=float, tl=250, repeat=5,
            #                              max_rule_num=10, debug=False, reg=r, max_col_num=m)

    print(res)
