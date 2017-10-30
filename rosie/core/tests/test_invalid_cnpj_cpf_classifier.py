from unittest import TestCase

import numpy as np
import pandas as pd

from rosie.core.classifiers import InvalidCnpjCpfClassifier


class TestInvalidCnpjCpfClassifier(TestCase):
    def setUp(self):
        self.dataset = pd.read_csv('rosie/core/tests/fixtures/invalid_cnpj_cpf_classifier.csv',
                                   dtype={'recipient_id': np.str})
        self.subject = InvalidCnpjCpfClassifier()

    def test_validation(self):
        self.assertFalse(self.subject.predict(self.dataset)[0], 'Test Valid CNPJ')
        self.assertTrue(self.subject.predict(self.dataset)[1], 'Test Invalid CNPJ')
        self.assertTrue(self.subject.predict(self.dataset)[2], 'Test None')
        self.assertFalse(self.subject.predict(self.dataset)[3], 'test_none_cnpj_cpf_abroad_is_valid')
        self.assertFalse(self.subject.predict(self.dataset)[4], 'test_valid_cnpj_cpf_abroad_is_valid')
        self.assertFalse(self.subject.predict(self.dataset)[5], 'test_invalid_cnpj_cpf_abroad_is_valid')
        self.assertFalse(self.subject.predict(self.dataset)[6], 'Test Valid CPF')
        self.assertTrue(self.subject.predict(self.dataset)[7], 'Test Invalid CPF')
        self.assertFalse(self.subject.predict(self.dataset)[8], 'Test Invalid Document Type')

    def test_fit(self):
        self.assertEqual(self.subject.fit(self.dataset), self.subject)

    def test_transform(self):
        self.assertEqual(self.subject.transform(), self.subject)
