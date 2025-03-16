from datetime import datetime

import numpy
import numpy as np
import pandas as pd
from numpy import log
from realkd.rules import Rule, AdditiveRuleEnsemble, CorrectiveRuleBoostingEstimator, loss_function, \
    RuleBoostingEstimator, XGBRuleEstimator, \
    GradientBoostingObjective
from pandas import qcut
from data_info import get_splits
from data_preprocess import preprocess_pd
from sklearn.model_selection import KFold


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
        train_y = train_y.replace(0, 1)
        test_x = pd.DataFrame(x[test_index], columns=labels)
        test_y = pd.Series(y[test_index])
        test_y = test_y.replace(0, 1)
        if method in ["b", "fc"]:
            rules = estimator.fit(train_x, train_y).rules_
        elif method == 'pp':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='weight').rules_
        elif method == 'pp_r':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='risk').rules_
        if method != "b" and method != "fc":
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


def evaluate(dataset_name, path, labels, feature_types, target, target_type=int, feature_map={}, loss='squared',
             repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    seeds = get_splits()[dataset_name]

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

        output = open(
            "../output20220820realkd_cv_poi/" + dataset_name + "realkd_col" + str(col) + str(datetime.now()) + ".txt",
            "a")
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        train_sr = pd.Series(train_target)
        train_sr = train_sr.replace(0, 1)
        train_sr = log(train_sr)
        test_sr = pd.Series(test_target)
        test_sr = test_sr.replace(0, 1)
        test_sr = log(test_sr)
        scores = {}
        # cross validation for boosting
        for r in regs:
            boosting = RuleBoostingEstimator(num_rules=max_rule_num,
                                             base_learner=XGBRuleEstimator(loss=loss, search='exhaustive',
                                                                           reg=r,
                                                                           search_params={
                                                                               'order': 'bestboundfirst',
                                                                               # 'bestboundfirst',
                                                                               'apx': 1.0,
                                                                               'max_depth': None,
                                                                               'discretization': qcut,
                                                                               'max_col_attr': col}))
            scores[r] = cv(train, train_target, boosting, labels, method='b', loss=loss)
        print('scores:', scores)
        # find best lambda
        reg = list(scores.keys())[0]
        for r in scores:
            if scores[r] < scores[reg]:
                reg = r
        # fit using boosting
        selected_regs.append(reg)
        boosting = RuleBoostingEstimator(num_rules=max_rule_num,
                                         base_learner=XGBRuleEstimator(loss=loss, search='exhaustive',
                                                                       reg=reg,
                                                                       search_params={
                                                                           'order': 'bestboundfirst',
                                                                           # 'bestboundfirst',
                                                                           'apx': 1.0,
                                                                           'max_depth': None,
                                                                           'discretization': qcut,
                                                                           'max_col_attr': col}))
        # boosting = RuleBoostingCorrectWeightEstimator(num_rules=max_rule_num, search='exhaustive', loss=loss,
        #                                               search_params={'max_col_attr': col},
        #                                               obj_func=GradientBoostingObjective)
        for r in regs:
            boosting.reg = r
            scores[r] = cv(train, train_target, fc_estimator, labels, method='fc', loss=loss)
        print('fc scores:', scores)
        # find best lambda
        reg = list(scores.keys())[0]
        for r in scores:
            if scores[r] < scores[reg]:
                reg = r
        selected_regs.append(reg)
        boosting.reg = reg
        try:
            b_rules = boosting.fit(train_df, train_sr)
            for r in range(1, max_rule_num + 1):
                ensemble = AdditiveRuleEnsemble(b_rules.rules_.members[:r])
                risk = sum(loss_func(ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in ensemble.members]) / 2 / n
                test_risk = sum(loss_func(ensemble(test_df), test_sr)) / len(test_sr)
                train_risk = sum(loss_func(ensemble(train_df), train_sr)) / n
                boosting_test_risk.append(test_risk)
                boosting_train_risk.append(train_risk)
                boosting_risk.append(risk)
                boosting_ensembles.append(str(ensemble))
                print(ensemble)
                print('risk', risk)
                print('train_risk', train_risk, 'test_risk', test_risk)
        except Exception as e:
            print('Error1: ', e)
        # cross validation for fc
        scores = {}
        for r in regs:
            fc_estimator.reg = r
            scores[r] = cv(train, train_target, fc_estimator, labels, method='fc', loss=loss)
        print('fc scores:', scores)
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
                fc_ensembles.append(str(fc_ensemble))
                print(fc_ensemble)
                print('risk', risk)
                print('train_risk', train_risk, 'test_risk', test_risk)
        except Exception as e:
            print('Error2: ', e)
        # # cross validation for pp (init)
        # scores = {}
        # for r in regs:
        #     fc_estimator.reg = r
        #     scores[r] = cv(train, train_target,
        #                    fc_estimator, labels, method='pp',
        #                    loss=loss)
        # print('scores:', scores)
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
        # print('scores:', scores)
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
                if i < len(boosting_risk):
                    output.write('boosting risk: ' + str(boosting_risk[i]) + '\n')
                    # output.write('boosting train score: ' + str(boosting_train_score[i]) + '\n')
                    # output.write('boosting test score: ' + str(boosting_test_score[i]) + '\n')
                    output.write('boosting train risk: ' + str(boosting_train_risk[i]) + '\n')
                    output.write('boosting test risk: ' + str(boosting_test_risk[i]) + '\n')
                    output.write(boosting_ensembles[i])
                if i < len(fc_risk):
                    output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
                    # output.write('fc train score: ' + str(fc_train_score[i]) + '\n')
                    # output.write('fc test score: ' + str(fc_test_score[i]) + '\n')
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


if __name__ == '__main__':
    res = {}
    # for col in [10]:
    #     try:
    #         res['titanic'] = evaluate('titanic', '../datasets/titanic/train.csv',
    #                                   ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    #                                   [int, str, float, int, int, float, str], 'Survived', target_type=int,
    #                                   feature_map={'Sex': {'male': 1, 'female': 0},
    #                                                'Embarked': {'S': 1, 'C': 2, 'Q': 3},
    #                                                'Survived': {'0': -1, '1': 1}}, loss='logistic',
    #                                   repeat=1, regs=[0.2],
    #                                   max_rule_num=10, col=col)
    #     except Exception as e:
    #         print("Error 1", e)

    res = {}
    numpy.seterr(divide='ignore')
    for col in [10]:
        res['covid_vic'] = evaluate('covid_vic', '../datasets/counting_data/COVID19 Data Viz LGA Data - lga.csv',
                                    ['population', 'active', 'cases', 'band'], [int, int, int, int],
                                    'new', target_type=int,
                                    repeat=5, max_rule_num=10,
                                    regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)
    for col in [10]:
        res['covid_world'] = evaluate('covid_world', '../datasets/counting_data/WHO-COVID-19-global-table-data.csv',
                                      ['Population', 'WHO_Region', 'Cases - cumulative total'], [int, int, int],
                                      'Cases - newly reported in last 24 hours', target_type=int,
                                      repeat=5, max_rule_num=10,
                                      regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)
    for col in [10]:
        res['vaccine'] = evaluate('vaccine', '../datasets/counting_data/vaccination-data.csv',
                                  ['Population', 'WHO_REGION', 'NUMBER_VACCINES_TYPES_USED'], [int, int, int],
                                  'TOTAL_VACCINATIONS', target_type=int,
                                  repeat=5, max_rule_num=10,
                                  regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)
    for col in [10]:
        res['covid'] = evaluate('covid', '../datasets/covid/COVID-19 Coronavirus.csv',
                                ['Population', 'Continent'],
                                [int, int],
                                'TotalDeaths/1M pop',
                                target_type=int, repeat=5, max_rule_num=10,
                                regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)
    for col in [10]:
        res['bicycle'] = evaluate('bicycle', '../datasets/bicycle_counts/Book1.csv',
                                  ['Day', 'High Temp (F)', 'Low Temp (F)', 'Precipitation'],
                                  [int, float, float, float],
                                  'Total',
                                  target_type=int, repeat=5, max_rule_num=10,
                                  regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)

    for col in [10]:
        res['austin-water'] = evaluate('austin-water',
                                       '../datasets/counting_data/AustinWater/' +
                                       'austin-water-wastewater-service-connection-count-by-zip-code-2.csv',
                                       ['PostCode', 'CustomerClass'], [int, int], 'ServiceConnections',
                                       target_type=int, repeat=5, max_rule_num=10,
                                       regs=[0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8], col=col)
