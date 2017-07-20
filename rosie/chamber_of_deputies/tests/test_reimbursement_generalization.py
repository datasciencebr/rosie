from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from rosie.chamber_of_deputies.classifiers.reimbursement_generalization import MealGeneralizationClassifier


class TestMealGeneralizationClassifier(TestCase):

    def setUp(self):
        self.model = 'rosie/chamber_of_deputies/classifiers/keras/model/weights.hdf5'
        self.dataset = pd.read_csv('rosie/chamber_of_deputies/tests/fixtures/generalization_reimbursements.csv',
                                   dtype={'document_id': np.str,
                          'applicant_id': np.str,
                          'year': np.str})
        self.subject = MealGeneralizationClassifier()
        self.subject.fit(self.model)
        self.prediction = self.subject.predict(self.dataset)

    def test_predict_true_when_generalized(self):
        assert_array_equal(np.repeat(True, 4),
                           self.prediction[[0, 2, 4, 5]])

    def test_predict_false_when_not_generalized(self):
        assert_array_equal(np.repeat(False, 2),
                          self.prediction[[1, 3]])
