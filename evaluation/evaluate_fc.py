from datetime import datetime

import pandas as pd
from realkd.rules import Rule, AdditiveRuleEnsemble, CorrectiveRuleBoostingEstimator, loss_function, \
    RuleBoostingEstimator, XGBRuleEstimator
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2
from pandas import qcut
from data_info import get_splits
from data_preprocess import preprocess_pd


def evaluate(dataset_name, path, labels, feature_types, target, target_type=int, feature_map={}, loss='squared',
             repeat=5, max_rule_num=5, reg=0, col=10):
    output = open(
        "../output20220206realkd/" + dataset_name + str(datetime.now()) + "realkd_col" + str(col) + "reg" + str(
            reg) + ".txt", "a")
    # 20211216
    seeds = get_splits()[dataset_name]
    func = skl_auc if loss == 'logistic' else skl_r2
    boosting_risk = []
    fc_risk = []
    pp_step_risk = []
    pp_final_risk = []
    boosting_train_score = []
    fc_train_score = []
    pp_step_train_score = []
    pp_final_train_score = []
    boosting_test_score = []
    fc_test_score = []
    pp_final_test_score = []
    pp_step_test_score = []
    loss_func = loss_function(loss)
    boosting_ensembles = []
    fc_ensembles = []
    pp_final_ensembles = []
    pp_step_ensembles = []
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
    fc_estimator = CorrectiveRuleBoostingEstimator(num_rules=max_rule_num, reg=reg, search='exhaustive', loss=loss,
                                                   search_params={'max_col_attr': col})
    for m in range(repeat):
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        train_sr = pd.Series(train_target)
        test_sr = pd.Series(test_target)
        try:
            b_rules = boosting.fit(train_df, train_sr)
            print(b_rules.rules_)
            for r in range(1, max_rule_num + 1):
                ensemble = AdditiveRuleEnsemble(b_rules.rules_.members[:r])
                risk = sum(loss_func(ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in ensemble.members]) / 2 / n
                boosting_risk.append(risk)
                train_score = func(train_target, ensemble(train_df))
                test_score = func(test_target, ensemble(test_df))
                boosting_train_score.append(train_score)
                boosting_test_score.append(test_score)
                boosting_ensembles.append(str(ensemble))

        except Exception as e:
            print('Error1: ', e)
        try:
            fc_rules = fc_estimator.fit(train_df, train_sr)
            print(fc_rules.rules_)
            for fc_ensemble in fc_estimator.history:
                risk = sum(loss_func(fc_ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
                fc_risk.append(risk)
                train_score = func(train_target, fc_ensemble(train_df))
                test_score = func(test_target, fc_ensemble(test_df))
                fc_train_score.append(train_score)
                fc_test_score.append(test_score)
                fc_ensembles.append(str(fc_ensemble))
        except Exception as e:
            print('Error2: ', e)
        try:
            pp_step_rules = fc_estimator.fc_post_processing_stepwise(train_df, train_sr, 'risk')
            print(pp_step_rules.rules_)
            for pp_ensemble in fc_estimator.history:
                risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in pp_ensemble.members]) / 2 / n
                pp_step_risk.append(risk)
                train_score = func(train_target, pp_ensemble(train_df))
                test_score = func(test_target, pp_ensemble(test_df))
                pp_step_train_score.append(train_score)
                pp_step_test_score.append(test_score)
                pp_step_ensembles.append(str(pp_ensemble))
        except Exception as e:
            print('Error3: ', e)
        try:
            pp_final_rules = fc_estimator.fit_final_post_processing(train_df, train_sr, 'risk')
            print(pp_final_rules.rules_)
            for pp_ensemble in fc_estimator.history:
                risk = sum(loss_func(pp_ensemble(train_df), train_sr)) / n + reg * sum(
                    [rule.y * rule.y for rule in pp_ensemble.members]) / 2 / n
                pp_final_risk.append(risk)
                train_score = func(train_target, pp_ensemble(train_df))
                test_score = func(test_target, pp_ensemble(test_df))
                pp_final_train_score.append(train_score)
                pp_final_test_score.append(test_score)
                pp_final_ensembles.append(str(pp_ensemble))
        except Exception as e:
            print('Error4: ', e)
        try:
            for i in range(max_rule_num):
                output.write('\n=======iteration ' + str(i) + '========\n')
                if i < len(boosting_risk):
                    output.write('boosting risk: ' + str(boosting_risk[i]) + '\n')
                    output.write('boosting train score: ' + str(boosting_train_score[i]) + '\n')
                    output.write('boosting test score: ' + str(boosting_test_score[i]) + '\n')
                    output.write(boosting_ensembles[i])
                if i < len(fc_risk):
                    output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
                    output.write('fc train score: ' + str(fc_train_score[i]) + '\n')
                    output.write('fc test score: ' + str(fc_test_score[i]) + '\n')
                    output.write(fc_ensembles[i])
                if i < len(pp_step_risk):
                    output.write('\npp step risk: ' + str(pp_step_risk[i]) + '\n')
                    output.write('pp step train score: ' + str(pp_step_train_score[i]) + '\n')
                    output.write('pp step test score: ' + str(pp_step_test_score[i]) + '\n')
                    output.write(pp_step_ensembles[i] + '\n')
                if i < len(pp_final_risk):
                    output.write('\npp final risk: ' + str(pp_final_risk[i]) + '\n')
                    output.write('pp final train score: ' + str(pp_final_train_score[i]) + '\n')
                    output.write('pp final test score: ' + str(pp_final_test_score[i]) + '\n')
                    output.write(pp_final_ensembles[i] + '\n')
        except Exception as e:
            print('Error5: ', e)
    output.close()
    return 0


if __name__ == '__main__':
    res = {}
    for col in [10]:
        for reg in [0]:
            print(col, reg)
            try:
                res['gdp'] = evaluate('gdp', '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
                                      ['GDP'], [int], 'Satisfaction', target_type=float, repeat=1,
                                      max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)
    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #             res['titanic'] = evaluate('titanic', '../datasets/titanic/train.csv',
    #                                   ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    #                                   [int, str, float, int, int, float, str], 'Survived', target_type=int,
    #                                   feature_map={'Sex': {'male': 1, 'female': 0},
    #                                                'Embarked': {'S': 1, 'C': 2, 'Q': 3},
    #                                                'Survived': {'0': -1, '1': 1}}, loss='logistic',
    #                                   repeat=1,
    #                                   max_rule_num=10, col=col, reg=reg)
    #         except Exception as e:
    #             print("Error 1", e)
    #
    #
    #
    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #
    #             res['wage'] = evaluate('wage', '../datasets/wages_demographics/wages.csv',
    #                                ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int], 'earn',
    #                                target_type=float,
    #                                feature_map={'sex': {'male': 1, 'female': 0},
    #                                             'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}},
    #                                repeat=1, max_rule_num=10, col=col, reg=reg)
    #         except Exception as e:
    #             print("Error 1", e)
    #
    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #             res['insurance'] = evaluate('insurance', '../datasets/insurance/insurance.csv',
    #                                     ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    #                                     [int, str, float, int, str, str], 'charges', target_type=float,
    #                                     feature_map={'sex': {'male': 1, 'female': 0},
    #                                                  'smoker': {'yes': 1, 'no': 0},
    #                                                  'region': {'southwest': 1, 'southeast': 2, 'northwest': 3,
    #                                                             'northeast': 4}},
    #                                     repeat=1, max_rule_num=10, col=col, reg=reg)
    #         except Exception as e:
    #             print("Error 1", e)

    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #             res['used_cars'] = evaluate('used_cars',
    #                                         '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
    #                                         ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
    #                                         target_type=float,
    #                                         repeat=1, max_rule_num=10, col=col, reg=reg)
    #         except Exception as e:
    #             print("Error 1", e)
    #
    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #             res['tic_tac_toe'] = evaluate('tic-tac-toe', '../datasets/tic_tac_toe/tic_tac_toe.csv',
    #                                           ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
    #                                           [str, str, str, str, str, str, str, str, str], 'V10', target_type=str,
    #                                           feature_map={'V1': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V2': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V3': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V4': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V5': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V6': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V7': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V8': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V9': {'x': 1, 'o': 2, 'b': 3},
    #                                                        'V10': {'positive': 1, 'negative': -1}},
    #                                           loss='logistic', repeat=1, max_rule_num=10, col=col, reg=reg)
    #         except Exception as e:
    #             print("Error 1", e)
    #
    # for col in [10]:
    #     for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
    #         try:
    #             res['boston'] = evaluate('boston', '../datasets/boston/boston_house_prices.csv',
    #                                      ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
    #                                       'B',
    #                                       'LSTAT'],
    #                                      [float, float, float, float, float, float, float, int, int, float, float,
    #                                       float],
    #                                      'MEDV',
    #                                      target_type=float, repeat=1, max_rule_num=10, col=col)
    #         except Exception as e:
    #             print("Error 1", e)
