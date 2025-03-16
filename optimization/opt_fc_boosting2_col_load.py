import copy
import sys
from datetime import datetime
from math import exp, log

import pandas as pd
from realkd.rules import AdditiveRuleEnsemble, loss_function
from sklearn.datasets import load_wine, load_iris, load_diabetes, load_breast_cancer

from boosting_col2_load import boosting_step2
from build_rule_ensemble import build_ensemble
from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd, preprocess_datasets
from fc_boosting_col2 import fully_corrective2
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2

from optimized_ensemble2_col import optimized_rule_ensemble2


def fc_opt_boosting(n, d, k, L, U, x, y, labels, reg=0, loss_func='squared', tl=100, f=None, left_most=0, debug=False,
                    max_col_num=10):
    ensembles = []
    weights = []
    lowers = []
    uppers = []
    risk = 1e10
    bnd = 1e10
    for i in range(1, k + 1):
        if f is not None:
            print('=====iteration ' + str(i) + '========')
            print('======boosting==========')
            f.write('=====iteration ' + str(i) + '========\n')
            f.write('======boosting==========\n')
        weights, lowers, uppers, yy, risk, bnd = boosting_step2(n, d, i, L, U, x, y, labels, reg=reg,
                                                                loss_func=loss_func,
                                                                tl=tl, init_w=weights, init_l=lowers, init_u=uppers,
                                                                f=f, left_most=left_most, debug=debug,
                                                                max_col_num=max_col_num)
        ensemble1 = build_ensemble(weights, lowers, uppers, L, U, labels)
        ensembles.append(ensemble1)

        # print('actual risk',sum(log(1+exp(-y_diff[j]*yy[j])) for j in range(len(y)))/n)
        if f is not None:
            # print('actual risk boosting: ', sum((yy[j] - y_diff[j]) ** 2 for j in range(n)) / n)
            f.write('boosting risk = ' + str(risk) + '\n')
            f.write('boosting result\n')
            f.write('weights: ' + str(weights) + '\n')
            f.write('lowers: ' + str(lowers) + '\n')
            f.write('uppers: ' + str(uppers) + '\n')
            f.write('ensemble 1: ' + str(ensemble1) + '\n\n')
            print(str(build_ensemble(weights, lowers, uppers, L, U, labels)))
            f.write('=======fully corrective=======\n')
            print('==========fully corrective=========')
        weights, yyy, risk_fc, bnd_fc = fully_corrective2(n, d, i, L, U, x, y, labels, weights, lowers, uppers, reg=reg,
                                                          loss_func=loss_func, tl=tl, f=f, max_col_num=max_col_num)
        ensemble3 = build_ensemble(weights, lowers, uppers, L, U, labels)
        ensembles.append(ensemble3)

        # print('actual risk',sum(log(1+exp(-y[j]*yyy[j])) for j in range(len(yyy)))/n)
        if f is not None:
            # print('actual risk fc boosting: ', sum((yyy[j] - y[j]) ** 2 for j in range(n)) / n)
            f.write('fc boosting risk = ' + str(risk_fc) + '\n')
            f.write('fc boosting bound= ' + str(bnd_fc) + '\n')
            f.write('weights: ' + str(weights) + '\n')
            f.write('lowers: ' + str(lowers) + '\n')
            f.write('uppers: ' + str(uppers) + '\n')
            f.write('ensemble3: ' + str(ensemble3) + '\n\n')
            f.write('==========optimization stage 2=========\n')
            print('========optimization stage 2=====')
        lowers, uppers, weights, yyy, risk, bnd = \
            optimized_rule_ensemble2(n, d, i, L, U, x, y, labels, reg=reg, loss_func=loss_func, tl=tl,
                                     init_w=weights, init_l=lowers, init_u=uppers, f=f, debug=debug,
                                     max_col_num=max_col_num)
        ensemble4 = build_ensemble(weights, lowers, uppers, L, U, labels)
        ensembles.append(ensemble4)
        if f is not None:
            f.write('opt s2 risk = ' + str(risk) + '\n')
            f.write('opt s2 bnd = ' + str(bnd) + '\n')
            f.write('weights: ' + str(weights) + '\n')
            f.write('lowers: ' + str(lowers) + '\n')
            f.write('uppers: ' + str(uppers) + '\n')
            f.write('ensemble 4: ' + str(ensemble4) + '\n\n\n')
        # y_diff = [y[j] - yyy[j] for j in range(len(y_diff))]
        print()
        # print('actual risk',sum(log(1+exp(-y[j]*yyy[j])) for j in range(len(yyy)))/n)
    return risk, ensembles, bnd


def evaluate_fc_boosting(dataset_name, load_method, cr='r', feature_map={},
                         loss='squared', tl=500, repeat=5, max_rule_num=5, debug=False, reg=0.0, max_col_num=10):
    seeds = get_splits()[dataset_name]
    k = max_rule_num
    func = skl_auc if cr == 'c' else skl_r2
    file = open(
        "../output20250524opt2/" + dataset_name + str(datetime.now()) + "_opt2_no_priority_sym_ind_950_reg" + str(
            reg) + "tl" + str(
            tl) + 'col' + str(max_col_num) + ".txt", "w")
    original_stdout = sys.stdout
    with open("../output20250524opt2/" + dataset_name +
              str(datetime.now()) + "_output_opt2_no_priority_sym_ind_950_reg" + str(reg) + "tl" + str(
        tl) + 'col' + str(max_col_num) + ".txt", "w") as f:
        # sys.stdout = f
        risks = []
        for m in range(repeat):
            file.write("======Dataset " + str(m) + "=======\n")
            train, test, train_target, test_target, L, U, d, n, labels = preprocess_datasets(load_method,
                                                                                             feature_map=feature_map,
                                                                                             random_seed=seeds[m])
            risk, ensembles, bnd = fc_opt_boosting(n, d, k, L, U, train, train_target, labels,
                                                   loss_func=loss, tl=tl, reg=reg, debug=debug,
                                                   f=file, max_col_num=max_col_num)
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

            risks.append(risk)
        file.close()
        print(dataset_name, 'opt, model2, fixed, no priority, symmetry, indicator, 950, tl=', tl, 'reg=', reg)
        sys.stdout = original_stdout
    print(dataset_name, risks)
    return risks


if __name__ == '__main__':
    res = {}
    for col in [10]:  # finished
        try:
            res['load_wine'] = evaluate_fc_boosting('load_wine', load_wine, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                        repeat=1, reg=0.1, loss='logistic',
                                        max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [10]:  # finished
        try:
            res['iris'] = evaluate_fc_boosting('iris', load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                   repeat=5, reg=0.1, loss='logistic',
                                   max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [5]:  # finished
        try:
            res['diabetes'] = evaluate_fc_boosting('load_diabetes', load_diabetes,
                                       repeat=5,
                                       max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)
    for col in [5]:
        try:
            res['breast'] = evaluate_fc_boosting('breast_cancer', load_breast_cancer,
                                     repeat=5, reg=0.1, loss='logistic',
                                     max_rule_num=10, max_col_num=col)
        except Exception as e:
            print("Error 1", e)