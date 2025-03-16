from datetime import datetime

import numpy as np
import pandas as pd
from realkd.rules import Rule, AdditiveRuleEnsemble, CorrectiveRuleBoostingEstimator, loss_function, \
    RuleBoostingEstimator, XGBRuleEstimator
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2
from pandas import qcut
from data_info import get_splits
from data_preprocess import preprocess_pd, preprocess_datasets, preprocess_gen
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_friedman1, make_friedman2, make_friedman3, \
    load_diabetes


# def cv(x, y, estimator, labels, loss='squared'):
#     kf = KFold(n_splits=5)
#     x = np.array(x)
#     y = np.array(y)
#     # validate = skl_auc if loss == 'logistic' else skl_r2
#     loss_func = loss_function(loss)
#     sum_scores = 0
#     for train_index, test_index in kf.split(x):
#         train_x = pd.DataFrame(x[train_index], columns=labels)
#         train_y = pd.Series(y[train_index])
#         test_x = pd.DataFrame(x[test_index], columns=labels)
#         test_y = pd.Series(y[test_index])
#         rules = estimator(train_x, train_y).rules_
#         # score = validate(test_y, rules(test_x))
#         score = sum(loss_func(rules(test_x), test_y)) / len(test_y)
#         print(rules)
#         print(score)
#         sum_scores += score
#     return sum_scores


def cv(x, y, estimator, labels, method, loss='squared'):
    kf = KFold(n_splits=5)
    x = np.array(x)
    y = np.array(y)
    loss_func = loss_function(loss)
    # validate = skl_auc if loss == 'logistic' else skl_r2
    sum_scores = 0
    for train_index, test_index in kf.split(x):
        train_x = pd.DataFrame(x[train_index], columns=labels)
        train_y = pd.Series(y[train_index])
        test_x = pd.DataFrame(x[test_index], columns=labels)
        test_y = pd.Series(y[test_index])
        if method in ["b", "fc"]:
            rules = estimator.fit(train_x, train_y).rules_
        elif method == 'pp':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='weight').rules_
        elif method == 'pp_r':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='risk').rules_
        if method != "b":
            for ensemble in estimator.history:
                score = sum(loss_func(ensemble(test_x), test_y)) / len(test_y)
                sum_scores += score
                print("n:", len(ensemble), "score: ", score)
        else:
            max_rule_num = len(rules)
            for r in range(1, max_rule_num + 1):
                ensemble = AdditiveRuleEnsemble(estimator.rules_.members[:r])
                score = sum(loss_func(ensemble(test_x), test_y)) / len(test_y)
                sum_scores += score
                print("n:", r, "score:", score)
    return sum_scores / 50


def evaluate(dataset_name, number, noise, d=4, loss='squared', test_size=0.2,
             repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    seeds = get_splits()[dataset_name]
    func = skl_auc if loss == 'logistic' else skl_r2

    for m in range(repeat):
        selected_regs = []
        boosting_risk = []
        fc_risk = []
        pp_r_risk = []
        pp_w_risk = []
        boosting_train_risk = []
        fc_train_risk = []
        pp_r_train_risk = []
        pp_w_train_risk = []
        boosting_test_risk = []
        fc_test_risk = []
        pp_r_test_risk = []
        pp_w_test_risk = []
        boosting_train_score = []
        fc_train_score = []
        pp_r_train_score = []
        pp_w_train_score = []
        boosting_test_score = []
        fc_test_score = []
        pp_w_test_score = []
        pp_r_test_score = []
        loss_func = loss_function(loss)
        boosting_ensembles = []
        fc_ensembles = []
        pp_w_ensembles = []
        pp_r_ensembles = []
        fc_estimator = CorrectiveRuleBoostingEstimator(num_rules=max_rule_num, search='exhaustive', loss=loss,
                                                       search_params={'max_col_attr': col})
        x, y, labels = gen_friedman(dataset_name, number, noise, seeds[m], d=d)
        output = open(
            "../output20220810realkd_cv/" + dataset_name + "_realkd_col" + str(col) + '_' + str(
                datetime.now()) + ".txt", "a")
        train, test, train_target, test_target, _, _, _, n = preprocess_gen(x, y, test_size=test_size,
                                                                            random_seed=seeds[m])
        print(train[0], train_target[0])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        train_sr = pd.Series(train_target)
        test_sr = pd.Series(test_target)
        scores = {}
        # # cross validation for boosting
        # for r in regs:
        #     boosting = RuleBoostingEstimator(num_rules=max_rule_num,
        #                                      base_learner=XGBRuleEstimator(loss=loss, search='exhaustive',
        #                                                                    reg=r,
        #                                                                    search_params={
        #                                                                        'order': 'bestboundfirst',
        #                                                                        # 'bestboundfirst',
        #                                                                        'apx': 1.0,
        #                                                                        'max_depth': None,
        #                                                                        'discretization': qcut,
        #                                                                        'max_col_attr': col}))
        #     scores[r] = cv(train, train_target, boosting, labels, method='b', loss=loss)
        # print(scores)
        # # find best lambda
        # reg = list(scores.keys())[0]
        # for r in scores:
        #     if scores[r] < scores[reg]:
        #         reg = r
        # # fit using boosting
        # selected_regs.append(reg)
        # boosting = RuleBoostingEstimator(num_rules=max_rule_num,
        #                                  base_learner=XGBRuleEstimator(loss=loss, search='exhaustive',
        #                                                                reg=reg,
        #                                                                search_params={
        #                                                                    'order': 'bestboundfirst',
        #                                                                    # 'bestboundfirst',
        #                                                                    'apx': 1.0,
        #                                                                    'max_depth': None,
        #                                                                    'discretization': qcut,
        #                                                                    'max_col_attr': col}))
        # try:
        #     b_rules = boosting.fit(train_df, train_sr)
        #     for r in range(1, max_rule_num + 1):
        #         ensemble = AdditiveRuleEnsemble(b_rules.rules_.members[:r])
        #         risk = sum(loss_func(ensemble(train_df), train_sr)) / n + reg * sum(
        #             [rule.y * rule.y for rule in ensemble.members]) / 2 / n
        #         test_risk = sum(loss_func(ensemble(test_df), test_sr)) / len(test_sr)
        #         train_risk = sum(loss_func(ensemble(train_df), train_sr)) / n
        #         boosting_test_risk.append(test_risk)
        #         boosting_train_risk.append(train_risk)
        #         boosting_risk.append(risk)
        #         train_score = func(train_target, ensemble(train_df))
        #         test_score = func(test_target, ensemble(test_df))
        #         boosting_train_score.append(train_score)
        #         boosting_test_score.append(test_score)
        #         boosting_ensembles.append(str(ensemble))
        #         print(ensemble)
        #         print('risk', risk)
        #         print('train_score', train_score, 'test_score', test_score)
        # except Exception as e:
        #     print('Error1: ', e)
        # cross validation for fc
        scores = {}
        for r in regs:
            fc_estimator.reg = r
            scores[r] = cv(train, train_target, fc_estimator, labels, method='fc', loss=loss)
        print(scores)
        # find best lambda
        reg = list(scores.keys())[0]
        for r in scores:
            if scores[r] < scores[reg]:
                reg = r
        selected_regs.append(reg)
        fc_estimator.reg = reg
        try:
            fc_rules = fc_estimator.fit(train_df, train_sr)
            print(fc_rules.rules_)
            for fc_ensemble in fc_estimator.history:
                risk = sum(loss_func(fc_ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
                test_risk = sum(loss_func(fc_ensemble(test_df), test_sr)) / len(test_sr)
                train_risk = sum(loss_func(fc_ensemble(train_df), train_sr)) / n
                fc_test_risk.append(test_risk)
                fc_train_risk.append(train_risk)
                fc_risk.append(risk)
                train_score = func(train_target, fc_ensemble(train_df))
                test_score = func(test_target, fc_ensemble(test_df))
                fc_train_score.append(train_score)
                fc_test_score.append(test_score)
                fc_ensembles.append(str(fc_ensemble))
                print(fc_ensemble)
                print('risk', risk)
                print('train_score', train_score, 'test_score', test_score)
        except Exception as e:
            print('Error2: ', e)
        # # cross validation for pp (init)
        # scores = {}
        # for r in regs:
        #     fc_estimator.reg = r
        #     scores[r] = cv(train, train_target,
        #                    fc_estimator, labels, method='pp',
        #                    loss=loss)
        # print(scores)
        # # find best lambda
        # reg = list(scores.keys())[0]
        # for r in scores:
        #     if scores[r] < scores[reg]:
        #         reg = r
        # selected_regs.append(reg)
        # fc_estimator.reg = reg
        # try:
        #     pp_final_rules = fc_estimator.fc_post_processing_stepwise(train_df, train_sr, pp_func='weight')
        #     print(pp_final_rules.rules_)
        #     for pp_ensemble in fc_estimator.history:
        #         risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n + reg * sum(
        #             [rule.y * rule.y for rule in pp_ensemble.members]) / 2 / n
        #         test_risk = sum(loss_func(pp_ensemble(test_df), test_sr)) / len(test_sr)
        #         train_risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n
        #         pp_w_test_risk.append(test_risk)
        #         pp_w_train_risk.append(train_risk)
        #         pp_w_risk.append(risk)
        #         train_score = func(train_target, pp_ensemble(train_df))
        #         test_score = func(test_target, pp_ensemble(test_df))
        #         pp_w_train_score.append(train_score)
        #         pp_w_test_score.append(test_score)
        #         pp_w_ensembles.append(str(pp_ensemble))
        #         print(pp_ensemble)
        #         print('risk', risk)
        #         print('train_score', train_score, 'test_score', test_score)
        # except Exception as e:
        #     print('Error4: ', e)
        # # cross validation for pp (risk)
        # scores = {}
        # for r in regs:
        #     fc_estimator.reg = r
        #     scores[r] = cv(train, train_target,
        #                    fc_estimator, labels, method='pp_r',
        #                    loss=loss)
        # print(scores)
        # # find best lambda
        # reg = list(scores.keys())[0]
        # for r in scores:
        #     if scores[r] < scores[reg]:
        #         reg = r
        # fc_estimator.reg = reg
        # selected_regs.append(reg)
        # try:
        #     pp_final_rules = fc_estimator.fc_post_processing_stepwise(train_df, train_sr, pp_func='risk')
        #     print(pp_final_rules.rules_)
        #     for pp_ensemble in fc_estimator.history:
        #         risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n + reg * sum(
        #             [rule.y * rule.y for rule in pp_ensemble.members]) / 2 / n
        #         test_risk = sum(loss_func(pp_ensemble(test_df), test_sr)) / len(test_sr)
        #         train_risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n
        #         pp_r_test_risk.append(test_risk)
        #         pp_r_train_risk.append(train_risk)
        #         pp_r_risk.append(risk)
        #         train_score = func(train_target, pp_ensemble(train_df))
        #         test_score = func(test_target, pp_ensemble(test_df))
        #         pp_r_train_score.append(train_score)
        #         pp_r_test_score.append(test_score)
        #         pp_r_ensembles.append(str(pp_ensemble))
        #         print(pp_ensemble)
        #         print('risk', risk)
        #         print('train_score', train_score, 'test_score', test_score)
        # except Exception as e:
        #     print('Error5: ', e)
        try:
            for i in range(max_rule_num):
                output.write('\n=======iteration ' + str(i) + '========\n')
                # if i < len(boosting_risk):
                #     output.write('boosting risk: ' + str(boosting_risk[i]) + '\n')
                #     output.write('boosting train score: ' + str(boosting_train_score[i]) + '\n')
                #     output.write('boosting test score: ' + str(boosting_test_score[i]) + '\n')
                #     output.write('boosting train risk: ' + str(boosting_train_risk[i]) + '\n')
                #     output.write('boosting test risk: ' + str(boosting_test_risk[i]) + '\n')
                #     output.write(boosting_ensembles[i])
                if i < len(fc_risk):
                    output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
                    output.write('fc train score: ' + str(fc_train_score[i]) + '\n')
                    output.write('fc test score: ' + str(fc_test_score[i]) + '\n')
                    output.write('fc train risk: ' + str(fc_train_risk[i]) + '\n')
                    output.write('fc test risk: ' + str(fc_test_risk[i]) + '\n')
                    output.write(fc_ensembles[i])
                # if i < len(pp_w_risk):
                #     output.write('\npp w risk: ' + str(pp_w_risk[i]) + '\n')
                #     output.write('pp w train score: ' + str(pp_w_train_score[i]) + '\n')
                #     output.write('pp w test score: ' + str(pp_w_test_score[i]) + '\n')
                #     output.write('pp w train risk: ' + str(pp_w_train_risk[i]) + '\n')
                #     output.write('pp w test risk: ' + str(pp_w_test_risk[i]) + '\n')
                #     output.write(pp_w_ensembles[i] + '\n')
                # if i < len(pp_r_risk):
                #     output.write('\npp final risk: ' + str(pp_r_risk[i]) + '\n')
                #     output.write('pp r train score: ' + str(pp_r_train_score[i]) + '\n')
                #     output.write('pp r test score: ' + str(pp_r_test_score[i]) + '\n')
                #     output.write('pp r train risk: ' + str(pp_r_train_risk[i]) + '\n')
                #     output.write('pp r test risk: ' + str(pp_r_test_risk[i]) + '\n')
                #     output.write(pp_r_ensembles[i] + '\n')
        except Exception as e:
            print('Error6: ', e)
        output.write(str(selected_regs))
        output.close()
    return 0


def gen_friedman(func_name, n, noise, random_seed, d=4):
    func_map = {'make_friedman1': make_friedman1, 'make_friedman2': make_friedman2, 'make_friedman3': make_friedman3, }
    if func_name == 'make_friedman1':
        x, y = func_map[func_name](n_samples=n, n_features=d, noise=noise, random_state=random_seed)
    else:
        x, y = func_map[func_name](n_samples=n, noise=noise, random_state=random_seed)
    labels = ['x' + str(i) for i in range(1, d + 1)]
    return x, y, labels


if __name__ == '__main__':
    res = {}
    for col in [10]:
        try:
            res['fried2'] = evaluate('make_friedman2', 10000, 0.1, test_size=0.8,
                                     repeat=5, regs=[0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                     max_rule_num=10, col=col)
        except Exception as e:
            print("Error 1", e)
    for col in [10]:
        try:
            res['fried3'] = evaluate('make_friedman3', 5000, 0.1, test_size=0.8,
                                     repeat=5, regs=[0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                     max_rule_num=10, col=col)
        except Exception as e:
            print("Error 1", e)
    for col in [4]:
        try:
            res['fried1'] = evaluate('make_friedman1', 2000, 0.1, d=10, test_size=0.8,
                                     repeat=5, regs=[0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                     max_rule_num=10, col=col)
        except Exception as e:
            print("Error 1", e)
