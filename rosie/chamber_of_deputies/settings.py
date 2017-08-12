from rosie.chamber_of_deputies.classifiers.election_expenses_classifier import ElectionExpensesClassifier
from rosie.chamber_of_deputies.classifiers.irregular_companies_classifier import IrregularCompaniesClassifier
from rosie.chamber_of_deputies.classifiers.meal_price_outlier_classifier import MealPriceOutlierClassifier
from rosie.chamber_of_deputies.classifiers.monthly_subquota_limit_classifier import MonthlySubquotaLimitClassifier
from rosie.chamber_of_deputies.classifiers.traveled_speeds_classifier import TraveledSpeedsClassifier
from rosie.chamber_of_deputies.classifiers.reimbursement_generalization import MealGeneralizationClassifier
from rosie.core.classifiers.invalid_cnpj_cpf_classifier import InvalidCnpjCpfClassifier


CLASSIFIERS = {
    'reimbursement_generalization': MealGeneralizationClassifier,
    'meal_price_outlier': MealPriceOutlierClassifier,
    'over_monthly_subquota_limit': MonthlySubquotaLimitClassifier,
    'suspicious_traveled_speed_day': TraveledSpeedsClassifier,
    'invalid_cnpj_cpf': InvalidCnpjCpfClassifier,
    'election_expenses': ElectionExpensesClassifier,
    'irregular_companies_classifier': IrregularCompaniesClassifier
}

UNIQUE_IDS = ['applicant_id', 'year', 'document_id']

SUPERVISED_MODEL = {'MealGeneralizationClassifier':
                    'https://drive.google.com/uc?export=download&id=0B6F2XOmMAf28dUFmUU92MWpxMFU'}
