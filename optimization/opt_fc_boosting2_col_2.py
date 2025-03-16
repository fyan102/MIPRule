import copy
import sys
from datetime import datetime
from math import exp, log

import pandas as pd
from realkd.rules import AdditiveRuleEnsemble, loss_function

from boosting_col2 import boosting_step2
from build_rule_ensemble import build_ensemble
from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd
from fc_boosting_col2 import fully_corrective2
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2

from optimized_ensemble2_col import optimized_rule_ensemble2


def fc_opt_boosting(n, d, k, L, U, x, y, labels, reg=0, loss_func='squared', tl=2500, f=None, left_most=0, debug=False,
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


def evaluate_fc_boosting(dataset_name, path, labels, feature_types, target, target_type, cr='r', feature_map={},
                         loss='squared', tl=2500, repeat=5, max_rule_num=5, debug=False, reg=0.0, max_col_num=10):
    seeds = get_splits()[dataset_name]
    k = max_rule_num
    func = skl_auc if cr == 'c' else skl_r2
    for m in range(repeat):
        file = open(
            "../output20251024opt2/" + dataset_name + "_opt2_no_priority_sym_ind_950_reg" + str(
                reg) + "tl" + str(
                tl) + 'col' + str(max_col_num) + 'rep' + str(m) + ".txt", "w")
        original_stdout = sys.stdout
        with open("../output20251024opt2/" + dataset_name + "_output_opt2_no_priority_sym_ind_950_reg" + str(
                reg) + "tl" + str(
            tl) + 'col' + str(max_col_num) + 'rep' + str(m) + ".txt", "w") as f:
            # sys.stdout = f
            risks = []

            file.write("======Dataset " + str(m) + "=======\n")
            train, test, train_target, test_target, L, U, d, n = preprocess_pd(path,
                                                                               labels,
                                                                               feature_types,
                                                                               target, target_type=target_type,
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
        # sys.stdout = original_stdout
    print(dataset_name, risks)
    return risks


if __name__ == '__main__':
    res = {}
    # for m in [10]:
    #     for r in [0]:
    #         res['gdp'] = evaluate_fc_boosting('gdp', '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
    #                                           ['GDP'], [int], 'Satisfaction', target_type=float, tl=250, repeat=5,
    #                                           max_rule_num=10, debug=False, reg=r, max_col_num=m)
    # for r in [0]:
    #     res['wage'] = evaluate_fc_boosting('wage', '../datasets/wages_demographics/wages.csv',
    #                                        ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int],
    #                                        'earn',
    #                                        target_type=float,
    #                                        feature_map={'sex': {'male': 1, 'female': 0},
    #                                                     'race': {'white': 1, 'black': 2, 'hispanic': 3,
    #                                                              'other': 4}},
    #                                        tl=250, repeat=5, max_rule_num=10, reg=r, max_col_num=m)
    # for r in [0.1]:
    #     res['titanic'] = evaluate_fc_boosting('titanic', '../datasets/titanic/train.csv',
    #                                           ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    #                                           [int, str, float, int, int, float, str], 'Survived', target_type=int,
    #                                           cr='c',
    #                                           feature_map={'Sex': {'male': 1, 'female': 0},
    #                                                        'Embarked': {'S': 1, 'C': 2, 'Q': 3},
    #                                                        'Survived': {'0': -1, '1': 1}}, loss='logistic', tl=250,
    #                                           repeat=5,
    #                                           max_rule_num=10, reg=r, max_col_num=m)
    # for r in [0]:
    #     res['insurance'] = evaluate_fc_boosting('insurance', '../datasets/insurance/insurance.csv',
    #                                             ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    #                                             [int, str, float, int, str, str], 'charges', target_type=float,
    #                                             feature_map={'sex': {'male': 1, 'female': 0},
    #                                                          'smoker': {'yes': 1, 'no': 0},
    #                                                          'region': {'southwest': 1, 'southeast': 2,
    #                                                                     'northwest': 3,
    #                                                                     'northeast': 4}},
    #                                             tl=250, repeat=5, max_rule_num=10, reg=r, max_col_num=m)
    # for r in [0]:
    #     res['used_cars'] = evaluate_fc_boosting('used_cars',
    #                                             '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
    #                                             ['count', 'km', 'year', 'powerPS'], [int, int, int, int],
    #                                             'avgPrice',
    #                                             target_type=float,
    #                                             tl=250, repeat=5, max_rule_num=10, reg=r, debug=False, max_col_num=m)
    # for r in [0.1]:
    #     res['tic_tac_toe'] = evaluate_fc_boosting('tic-tac-toe', '../datasets/tic_tac_toe/tic_tac_toe.csv',
    #                                               ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
    #                                               [str, str, str, str, str, str, str, str, str], 'V10',
    #                                               target_type=str,
    #                                               cr='c',
    #                                               feature_map={'V1': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V2': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V3': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V4': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V5': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V6': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V7': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V8': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V9': {'x': 1, 'o': 2, 'b': 3},
    #                                                            'V10': {'positive': 1, 'negative': -1}},
    #                                               loss='logistic', tl=250, repeat=5, max_rule_num=10, reg=r,
    #                                               max_col_num=m)
    # for r in [0]:
    #     res['boston'] = evaluate_fc_boosting('boston', '../datasets/boston/boston_house_prices.csv',
    #                                          ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    #                                           'PTRATIO',
    #                                           'B',
    #                                           'LSTAT'],
    #                                          [float, float, float, float, float, float, float, int, int, float,
    #                                           float,
    #                                           float],
    #                                          'MEDV', max_col_num=4,
    #                                          target_type=float, tl=250, repeat=5, max_rule_num=10, reg=r)
    # for col in [10]:  # finished
    #     try:
    #         res['world_happiness_indicator'] = evaluate_fc_boosting('world_happiness_indicator',
    #                                                                 '../datasets/world_happiness_indicator/2019.csv',
    #                                                                 ['GDP per capita', 'Social support',
    #                                                                  'Healthy life expectancy',
    #                                                                  'Freedom to make life choices',
    #                                                                  'Generosity', 'Perceptions of corruption'],
    #                                                                 [float, float, float, float, float, float, ],
    #                                                                 'Score',
    #                                                                 target_type=float, tl=250, reg=0,
    #                                                                 repeat=5, max_rule_num=10, max_col_num=col)
    #     except Exception as e:
    #         print("Error 1", e)
    # for col in [4]:
    #     try:
    #         res['Demographics'] = evaluate_fc_boosting('Demographics', '../datasets/Demographics/Demographics1.csv',
    #                                                    ['Sex', 'Marital', 'Age', 'Edu', 'Occupation', 'LivingYears',
    #                                                     'Persons',
    #                                                     'PersonsUnder18', 'HouseholderStatus',
    #                                                     'TypeOfHome', 'Ethnic', 'Language'],
    #                                                    [str, str, int, int, str, int, int, int, str, str, str, str],
    #                                                    'AnnualIncome',
    #                                                    target_type=int,
    #                                                    feature_map={'Sex': {' Male': 1, ' Female': 0},
    #                                                                 'Marital': {' Married': 1, '': 0,
    #                                                                             ' Single, never married': 2,
    #                                                                             ' Divorced or separated': 3,
    #                                                                             ' Living together, not married': 4,
    #                                                                             ' Widowed': 5},
    #                                                                 'Occupation': {'': 0, ' Homemaker': 1,
    #                                                                                ' Professional/Managerial': 2,
    #                                                                                ' Student, HS or College': 3,
    #                                                                                ' Retired': 4, ' Unemployed': 5,
    #                                                                                ' Factory Worker/Laborer/Driver': 6,
    #                                                                                ' Sales Worker': 7,
    #                                                                                ' Clerical/Service Worker': 8,
    #                                                                                ' Military': 9},
    #                                                                 'HouseholderStatus': {'': 0, ' Own': 1, ' Rent': 2,
    #                                                                                       ' Live with Parents/Family': 3},
    #                                                                 'TypeOfHome': {'': 0, ' House': 1,
    #                                                                                ' Apartment': 2,
    #                                                                                ' Condominium': 3,
    #                                                                                ' Mobile Home': 4, ' Other': 5, },
    #                                                                 'Ethnic': {'': 0, ' White': 1,
    #                                                                            ' Hispanic': 2,
    #                                                                            ' Asian': 3,
    #                                                                            ' Black': 4, ' East Indian': 5,
    #                                                                            ' Pacific Islander': 6,
    #                                                                            ' American Indian': 7,
    #                                                                            ' Other': 8, },
    #                                                                 'Language': {'': 0, ' English': 1, ' Spanish': 2,
    #                                                                              ' Other': 3, }
    #                                                                 }, tl=250, reg=0,
    #                                                    repeat=5, max_rule_num=10, max_col_num=col)
    #     except Exception as e:
    #         print("Error 1", e)

    for col in [4]:
        res['IBM_HR'] = evaluate_fc_boosting('IBM_HR', '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                             ["Age", 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                                              'Education',
                                              'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                                              'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                                              'MaritalStatus',
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
                                                                             'Technical Degree': 4,
                                                                             'Human Resources': 5,
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

    for col in [5]:
        try:
            res['telco_churn'] = evaluate_fc_boosting('telco_churn',
                                                      '../datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                                                      ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                                       'PhoneService', 'MultipleLines', 'InternetService',
                                                       'OnlineSecurity',
                                                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                       'StreamingMovies', 'Contract', 'PaperlessBilling',
                                                       'PaymentMethod',
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
    #
    # for col in [4]:
    #     try:
    #         res['mobile_prices'] = evaluate_fc_boosting('mobile_prices',
    #                                                     '../datasets/mobile_prices/train.csv',
    #                                                     ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    #                                                      'four_g',
    #                                                      'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
    #                                                      'px_height',
    #                                                      'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
    #                                                      'touch_screen',
    #                                                      'wifi'],
    #                                                     [int, int, float, int, int, int, int, float, int, int, int, int,
    #                                                      int, int,
    #                                                      int,
    #                                                      int, int, int, int, int, ], 'price_range',
    #                                                     target_type=int,
    #                                                     tl=250, reg=0,
    #                                                     repeat=5, max_rule_num=10, max_col_num=col)
    #     except Exception as e:
    #         print("Error 1", e)
    for col in [5]:
        try:
            res['GenderRecognition'] = evaluate_fc_boosting('GenderRecognition',
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
    # for m in [10]:
    #     for r in [0]:
    #         res['social_media'] = evaluate_fc_boosting('social_media', '../datasets/social_media/social-media.csv',
    #                                                    ['UsageDuraiton', 'Age'], [int, int], 'TotalLikes',
    #                                                    target_type=float, tl=250, repeat=5,
    #                                                    max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['salary'] = evaluate_fc_boosting('salary', '../datasets/salary/Salary_dataset.csv',
    #                                              ['YearsExperience'], [float], 'Salary',
    #                                              target_type=float, tl=250, repeat=5,
    #                                              max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['student_marks'] = evaluate_fc_boosting('student_marks', '../datasets/student_marks/Student_Marks.csv',
    #                                                     ['number_courses', 'time_study'], [int, float], 'Marks',
    #                                                     target_type=float, tl=250, repeat=5,
    #                                                     max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['study_time'] = evaluate_fc_boosting('study_time', '../datasets/study_time/score_updated.csv',
    #                                                  ['Hours'], [float], 'Scores',
    #                                                  target_type=float, tl=250, repeat=5,
    #                                                  max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['income'] = evaluate_fc_boosting('income', '../datasets/income/multiple_linear_regression_dataset.csv',
    #                                              ['age', 'experience'], [int, int], 'income',
    #                                              target_type=float, tl=250, repeat=5,
    #                                              max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['headbrain'] = evaluate_fc_boosting('headbrain', '../datasets/headbrain/headbrain.csv',
    #                                                 ['Gender', 'Age Range', 'Head Size(cm^3)'], [int, int, int],
    #                                                 'Brain Weight(grams)',
    #                                                 target_type=float, tl=250, repeat=5,
    #                                                 max_rule_num=10, debug=False, reg=r, max_col_num=m)
    #         res['fitness'] = evaluate_fc_boosting('fitness', '../datasets/fitness/data.csv',
    #                                               ['Duration', 'Pulse', 'Maxpulse'], [int, int, int], 'Calories',
    #                                               target_type=float, tl=250, repeat=5,
    #                                               max_rule_num=10, debug=False, reg=r, max_col_num=m)
    # res[''] = evaluate_boosting2('', '../datasets/',
    #                              [''], [], '',
    #                              target_type=float, tl=250, repeat=5,
    #                              max_rule_num=10, debug=False, reg=r, max_col_num=m)
    for col in [10]:
        res['liver'] = evaluate_fc_boosting('liver', '../datasets/liver/liver.csv',
                                            ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks'],
                                            [int, int, int, int, int, float], 'selector', target_type=int,
                                            tl=500, repeat=5, loss='logistic',
                                            max_rule_num=10, debug=False, reg=0.1, cr='c', max_col_num=col)
    for col in [5]:
        res['magic'] = evaluate_fc_boosting('magic', '../datasets/magic/magic04.csv',
                                            ['fLen1t-1', 'fWidt-1', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Lon1',
                                             'fM3Trans', 'fAlp-1a', 'fDist'],
                                            [float, float, float, float, float, float, float, float, float, float],
                                            'class',
                                            target_type=int, tl=500, repeat=5, loss='logistic',
                                            max_rule_num=10, debug=False, reg=0.1, cr='c', max_col_num=col)
        res['adult'] = evaluate_fc_boosting('adult', '../datasets/adult/adult.csv',
                                            ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                                             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                             'hours-per-week'],
                                            [int, str, int, int, str, str, str, int, int, int, int],
                                            'output', target_type=int, tl=500, repeat=5, loss='logistic',
                                            max_rule_num=10, debug=False, reg=0.1, cr='c', max_col_num=col)

    for col in [4]:
        res['digits5'] = evaluate_fc_boosting('digits5', '../datasets/digits/digits.csv',
                                              ['pixel_' + str(i) + '_' + str(j) for j in range(8) for i in range(8)],
                                              [int] * 64,
                                              'target',
                                              feature_map={
                                                  'target': {'5': 1, '0': -1, '1': -1, '2': -1, '3': -1, '4': -1,
                                                             '6': -1,
                                                             '7': -1, '8': -1,
                                                             '9': -1}},
                                              target_type=int, tl=500, repeat=5, loss='logistic',
                                              max_rule_num=10, debug=False, reg=0.1, cr='c', max_col_num=col)
