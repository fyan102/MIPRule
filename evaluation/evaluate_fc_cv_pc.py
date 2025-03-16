from datetime import datetime

import numpy as np
import pandas as pd
from realkd.rules import Rule, AdditiveRuleEnsemble, CorrectiveRuleBoostingEstimator, loss_function, \
    RuleBoostingEstimator, XGBRuleEstimator
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2
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
        test_x = pd.DataFrame(x[test_index], columns=labels)
        test_y = pd.Series(y[test_index])
        if method in ["b", "fc"]:
            rules = estimator.fit(train_x, train_y).rules_
        elif method == 'pc':
            rules = estimator.fit_pc(train_x, train_y).rules_
        elif method == 'pp':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='weight').rules_
        elif method == 'pp_r':
            rules = estimator.fc_post_processing_stepwise(train_x, train_y, pp_func='risk').rules_
        if method != "b" and method != 'pc':
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
        output = open(
            "../output20220329realkd_cv/" + dataset_name + "realkd_col" + str(col) + str(datetime.now()) + ".txt", "a")
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
        scores = {}
        # cross validation for boosting
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
        # print('scores:', scores)
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
        # # cross validation for fc
        scores = {}
        for r in regs:
            fc_estimator.reg = r
            scores[r] = cv(train, train_target, fc_estimator, labels, method='pc', loss=loss)
        print('scores:', scores)
        # find best lambda
        reg = list(scores.keys())[0]
        for r in scores:
            if scores[r] < scores[reg]:
                reg = r
        selected_regs.append(reg)
        fc_estimator.reg = reg
        # try:
        fc_rules = fc_estimator.fit_pc(train_df, train_sr)
        print(fc_rules.rules_)
        for r in range(1, max_rule_num + 1):
            fc_ensemble = AdditiveRuleEnsemble(fc_rules.rules_.members[:r])
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
        # except Exception as e:
        #     print('Error2: ', e)
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
                # if i < len(boosting_risk):
                #     output.write('boosting risk: ' + str(boosting_risk[i]) + '\n')
                #     output.write('boosting train score: ' + str(boosting_train_score[i]) + '\n')
                #     output.write('boosting test score: ' + str(boosting_test_score[i]) + '\n')
                #     output.write('boosting train risk: ' + str(boosting_train_risk[i]) + '\n')
                #     output.write('boosting test risk: ' + str(boosting_test_risk[i]) + '\n')
                #     output.write(boosting_ensembles[i])
                if i < len(fc_risk):
                    output.write('\npc risk: ' + str(fc_risk[i]) + '\n')
                    output.write('pc train score: ' + str(fc_train_score[i]) + '\n')
                    output.write('pc test score: ' + str(fc_test_score[i]) + '\n')
                    output.write('pc train risk: ' + str(fc_train_risk[i]) + '\n')
                    output.write('pc test risk: ' + str(fc_test_risk[i]) + '\n')
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
    # for col in [20]:
    #     res['gdp'] = evaluate('gdp', '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
    #                           ['GDP'], [int], 'Satisfaction', target_type=float, repeat=5,
    #                           max_rule_num=10, col=col)
    for col in [10]:

            res['titanic'] = evaluate('titanic', '../datasets/titanic/train.csv',
                                      ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                                      [int, str, float, int, int, float, str], 'Survived', target_type=int,
                                      feature_map={'Sex': {'male': 1, 'female': 0},
                                                   'Embarked': {'S': 1, 'C': 2, 'Q': 3},
                                                   'Survived': {'0': -1, '1': 1}}, loss='logistic',
                                      repeat=5, regs=[0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                      max_rule_num=10, col=col)

    #
    # for col in [10]:
    #     try:
    #         res['wage'] = evaluate('wage', '../datasets/wages_demographics/wages.csv',
    #                                ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int], 'earn',
    #                                target_type=float,
    #                                feature_map={'sex': {'male': 1, 'female': 0},
    #                                             'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}},
    #                                repeat=5, max_rule_num=10, col=col)
    #     except Exception as e:
    #         print("Error 1", e)
    #
    # for col in [10]:
    #     try:
    #         res['insurance'] = evaluate('insurance', '../datasets/insurance/insurance.csv',
    #                                     ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    #                                     [int, str, float, int, str, str], 'charges', target_type=float,
    #                                     feature_map={'sex': {'male': 1, 'female': 0},
    #                                                  'smoker': {'yes': 1, 'no': 0},
    #                                                  'region': {'southwest': 1, 'southeast': 2, 'northwest': 3,
    #                                                             'northeast': 4}},
    #                                     repeat=5, max_rule_num=10, col=col)
    #     except Exception as e:
    #         print("Error 1", e)
    #
    # for col in [10]:
    #     try:
    #         res['used_cars'] = evaluate('used_cars',
    #                                     '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
    #                                     ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
    #                                     target_type=float,
    #                                     repeat=5, max_rule_num=10, col=col)
    #     except Exception as e:
    #         print("Error 1", e)
    #
    for col in [10]:
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
                                          regs=[0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                          loss='logistic', repeat=5, max_rule_num=10, col=col)
        except Exception as e:
            print("Error 1", e)

    # for col in [4]:
    #     try:
    #         res['boston'] = evaluate('boston', '../datasets/boston/boston_house_prices.csv',
    #                                  ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
    #                                   'B',
    #                                   'LSTAT'],
    #                                  [float, float, float, float, float, float, float, int, int, float, float,
    #                                   float],
    #                                  'MEDV',
    #                                  target_type=float, repeat=5, max_rule_num=10, col=col)
    #     except Exception as e:
    #         print("Error 1", e)
    for col in [3]:
        res['IBM_HR'] = evaluate('IBM_HR', '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                 ["Age", 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
                                  'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                                  'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                                  'MonthlyIncome',
                                  'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                                  'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                                  'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                                  'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
                                 [int, str, int, str, int, int, str, int, str, int, int, int, str, int, str, int, int,
                                  int, str, int, int, int, int, int, int, int, int, int, int, int, int],
                                 'Attrition', target_type=str,
                                 feature_map={"BusinessTravel":{'Travel_Rarely':1,'Travel_Frequently':2,'Non-Travel':3},
                                     "Attrition": {'Yes': 1, 'No': -1},
                                              'Department': {'Sales': 1, 'Research & Development': 2,
                                                             'Human Resources': 3},
                                              'EducationField': {'Life Sciences': 1, 'Medical': 2, 'Marketing': 3,
                                                                 'Technical Degree': 4, 'Human Resources': 5,
                                                                 'Other': 6},
                                              'Gender': {'Male': 1, 'Female': 0},
                                              'JobRole': {'Sales Executive': 1, 'Research Scientist': 2,
                                                          'Laboratory Technician': 3,
                                                          'Manufacturing Director': 4, 'Healthcare Representative': 5,
                                                          'Manager': 6, 'Human Resources': 7, 'Research Director': 8,
                                                          'Sales Representative': 9},
                                              'MaritalStatus': {'Single': 1, 'Married': 2, 'Divorced': 3},
                                              'OverTime': {'Yes': 1, 'No': 0}, },
                                 loss='logistic', repeat=5, regs=[0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                 max_rule_num=10, col=col)
    for col in [3]:
        try:
            res['GenderRecognition'] = evaluate('GenderRecognition', '../datasets/GenderRecognition/voice.csv',
                                                ["meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt",
                                                 "sp.ent", "sfm", "mode", "centroid", "meanfun", "minfun", "maxfun",
                                                 "meandom", "mindom", "maxdom", "dfrange", "modindx"],
                                                [float, float, float, float, float, float, float, float, float, float,
                                                 float, float, float, float, float, float, float, float, float, float],
                                                "label", target_type=str,
                                                feature_map={"label": {'male': 1, 'female': -1}}, loss='logistic',
                                                repeat=5, regs=[0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                                max_rule_num=10, col=col)
        except Exception as e:
            print("Error 1", e)
    for col in [4]:
        try:
            res['telco_churn'] = evaluate('telco_churn',
                                          '../datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                                          ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                           'MonthlyCharges', 'TotalCharges', ],
                                          [str, int, str, str, int, str, str, str, str, str, str, str, str, str,
                                           str, str, str, float, float],
                                          'Churn',
                                          target_type=str,
                                          feature_map={'gender': {'Male': 1, 'Female': 0},
                                                       'Partner': {'Yes': 1, 'No': 0},
                                                       'Dependents': {'Yes': 1, 'No': 0},
                                                       'PhoneService': {'Yes': 1, 'No': 0},
                                                       'MultipleLines': {'Yes': 1, 'No': 2, 'No phone service': 3},
                                                       'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 3},
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
                                                       'PaymentMethod': {'Electronic check': 1, 'Mailed check': 2,
                                                                         'Bank transfer (automatic)': 3,
                                                                         'Credit card (automatic)': 4},
                                                       'Churn': {'Yes': 1, 'No': -1},
                                                       }, regs=[0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16],
                                          repeat=5, max_rule_num=10, loss='logistic', col=col)
        except Exception as e:
            print("Error 1", e)

