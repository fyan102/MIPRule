
from realkd.logic import KeyValueProposition, Constraint, Conjunction
from realkd.rules import Rule, AdditiveRuleEnsemble


def build_ensemble(weights, lowers, uppers, L, U, labels):
    k = len(weights)
    d = len(lowers[0])
    rules = []
    for i in range(k):
        rules.append(build_one_rule(weights[i], lowers[i], uppers[i], L, U, labels))
    rule_ensemble = AdditiveRuleEnsemble(rules)
    return rule_ensemble


def build_one_rule(weight, lowers, uppers, L, U, labels):
    d = len(lowers)
    propositions = []
    for j in range(d):
        if lowers[j] != L[j]:
            propositions.append(KeyValueProposition(labels[j], Constraint.greater_equals(lowers[j])))
        if uppers[j] != U[j]:
            propositions.append(KeyValueProposition(labels[j], Constraint.less_equals(uppers[j])))
    return Rule(Conjunction(propositions), weight, 0)


def add_one_rule(ensemble: AdditiveRuleEnsemble, rule):
    return AdditiveRuleEnsemble(ensemble.members + [rule])
