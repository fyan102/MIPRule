from datetime import datetime

import pandas as pd
from realkd.logic import KeyValueProposition, Constraint, Conjunction
from realkd.rules import Rule, AdditiveRuleEnsemble

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd
from optimization.optimized_ensemble import optimized_rule_ensemble, boosting_optimized_rule_ensemble
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2


def evaluate(dataset_name, path, labels, feature_types, target, target_type=int, cr='r', feature_map={}, loss='squared',
             tl=100, repeat=5, max_rule_num=5, boosting=False, reg=0):
    output = open("../output20220725realkd_cv/" + dataset_name + str(datetime.now()) + ".txt", "a")
    seeds = get_splits()[dataset_name]
    train_roc = []
    test_roc = []
    func = skl_auc if cr == 'c' else skl_r2
    optimize_func = boosting_optimized_rule_ensemble if boosting else optimized_rule_ensemble
    boosting_risk = [0] * max_rule_num
    opt_risk = [0] * max_rule_num
    for m in range(repeat):
        tr_roc = []
        te_roc = []
        train, test, train_target, test_target, L, U, d, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        for k in range(1, max_rule_num + 1):
            opt_res, boosting_rules, b_risk = optimize_func(n, d, k, L, U, train, train_target,
                                                            labels, loss_func=loss, tl=tl * k, f=output)
            train_df = pd.DataFrame(train, columns=labels)
            test_df = pd.DataFrame(test, columns=labels)
            rules = []
            print(opt_res)
            ws, ls, us, risk = opt_res
            opt_risk[k - 1] += risk
            boosting_risk[k - 1] += b_risk
            output.write('opt risk: ' + str(risk) + '\n')
            print('opt risk: ' + str(risk))
            for i in range(k):
                propositions = []
                for j in range(d):
                    if ls[i][j] != L[j]:
                        propositions.append(KeyValueProposition(labels[j], Constraint.greater_equals(ls[i][j])))
                    if us[i][j] != U[j]:
                        propositions.append(KeyValueProposition(labels[j], Constraint.less_equals(us[i][j])))
                rules.append(Rule(Conjunction(propositions), ws[i], 0))
            rule_ensemble = AdditiveRuleEnsemble(rules)
            if boosting_rules is not None:
                boosting_test_score = func(test_target, boosting_rules(test_df))
                boosting_train_score = func(train_target, boosting_rules(train_df))
                print("boosting test     train")
                print(boosting_test_score, boosting_train_score)
                output.write("boosting test     train\n")
                output.write(str(boosting_test_score) + " " + str(boosting_train_score) + "\n")
            test_score = func(test_target, rule_ensemble(test_df))
            train_score = func(train_target, rule_ensemble(train_df))
            print(test_score, train_score)
            output.write('opt test       train\n')
            output.write(str(test_score) + " " + str(train_score) + "\n")
            tr_roc.append(train_score)
            te_roc.append(test_score)
        train_roc.append(tr_roc)
        test_roc.append(te_roc)
    print(train_roc, test_roc)
    output.write(str(train_roc) + " " + str(test_roc) + "\n")
    roc_train = []
    roc_test = []
    for i in range(max_rule_num):
        sum_tr = 0
        sum_te = 0
        for j in range(repeat):
            sum_tr += train_roc[j][i]
            sum_te += test_roc[j][i]
        roc_train += [sum_tr / repeat]
        roc_test += [sum_te / repeat]
    # print(roc_train)
    # print(roc_test)
    opt_risk = [x / 5 for x in opt_risk]
    boosting_risk = [x / 5 for x in boosting_risk]
    output.write('opt_risk' + str(opt_risk) + '\n')
    output.write('boosting_risk' + str(boosting_risk) + '\n')
    output.write(str(roc_train) + " " + str(roc_test) + "\n")
    output.close()
    return roc_train, roc_test


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
    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:
                res['titanic'] = evaluate('titanic', '../datasets/titanic/train.csv',
                                          ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                                          [int, str, float, int, int, float, str], 'Survived', target_type=int,
                                          feature_map={'Sex': {'male': 1, 'female': 0},
                                                       'Embarked': {'S': 1, 'C': 2, 'Q': 3},
                                                       'Survived': {'0': -1, '1': 1}}, loss='logistic',
                                          repeat=1,
                                          max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)

    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:

                res['wage'] = evaluate('wage', '../datasets/wages_demographics/wages.csv',
                                       ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int], 'earn',
                                       target_type=float,
                                       feature_map={'sex': {'male': 1, 'female': 0},
                                                    'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}},
                                       repeat=1, max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)

    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:
                res['insurance'] = evaluate('insurance', '../datasets/insurance/insurance.csv',
                                            ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                                            [int, str, float, int, str, str], 'charges', target_type=float,
                                            feature_map={'sex': {'male': 1, 'female': 0},
                                                         'smoker': {'yes': 1, 'no': 0},
                                                         'region': {'southwest': 1, 'southeast': 2, 'northwest': 3,
                                                                    'northeast': 4}},
                                            repeat=1, max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)

    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:
                res['used_cars'] = evaluate('used_cars',
                                            '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
                                            ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
                                            target_type=float,
                                            repeat=1, max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)

    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:
                res['tic_tac_toe'] = evaluate('tic-tac-toe', '../datasets/tic_tac_toe/tic_tac_toe.csv',
                                              ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
                                              [str, str, str, str, str, str, str, str, str], 'V10', target_type=str,
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
                                              loss='logistic', repeat=1, max_rule_num=10, col=col, reg=reg)
            except Exception as e:
                print("Error 1", e)

    for col in [10]:
        for reg in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            try:
                res['boston'] = evaluate('boston', '../datasets/boston/boston_house_prices.csv',
                                         ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                          'B',
                                          'LSTAT'],
                                         [float, float, float, float, float, float, float, int, int, float, float,
                                          float],
                                         'MEDV',
                                         target_type=float, repeat=1, max_rule_num=10, col=col)
            except Exception as e:
                print("Error 1", e)
