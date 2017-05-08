import os
from unittest import TestCase

import numpy as np
import pandas as pd

from rosie.chamber_of_deputies.classifiers.irregular_companies_classifier import IrregularCompaniesClassifier


class TestIrregularCompaniesClassifier(TestCase):
    def setUp(self):
        from . import get_fixtures_dir
        FIXTURES_DIR = get_fixtures_dir()
        the_file = os.path.join(FIXTURES_DIR, 'irregular_companies_classifier.csv')
        self.dataset = pd.read_csv(the_file,
                                   dtype={'cnpj': np.str})
        self.subject = IrregularCompaniesClassifier()

    def test_is_regular_company(self):
        self.assertEqual(self.subject.predict(self.dataset)[0], False)

    def test_is_irregular_company_BAIXADA(self):
        self.assertEqual(self.subject.predict(self.dataset)[1], True)

    def test_is_irregular_company_NULA(self):
        self.assertEqual(self.subject.predict(self.dataset)[2], True)

    def test_is_irregular_company_INAPTA(self):
        self.assertEqual(self.subject.predict(self.dataset)[3], True)

    def test_is_irregular_company_SUSPENSA(self):
        self.assertEqual(self.subject.predict(self.dataset)[4], True)

    def test_is_valid_reimbursement_based_on_date(self):
        self.assertEqual(self.subject.predict(self.dataset)[5], False)

    def test_fit(self):
        self.assertEqual(self.subject.fit(self.dataset), self.subject)

    def test_transform(self):
        self.assertEqual(self.subject.transform(), self.subject)
