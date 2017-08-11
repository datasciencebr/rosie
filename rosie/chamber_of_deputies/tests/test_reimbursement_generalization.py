from unittest import TestCase
import os
import tempfile
import shutil
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
        # Fit the classifier using the robust ML model
        self.subject.fit(self.model)
        # Use the robust ML model to predict reimbursements
        self.prediction = self.subject.predict(self.dataset)

        # Creates a fake model for test
        self.directories = ['rosie/chamber_of_deputies/classifiers/keras/dataset/',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/training',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/training/positive/',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/training/negative/',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/validation/',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/validation/positive/',
                'rosie/chamber_of_deputies/classifiers/keras/dataset/validation/negative/',
                'rosie/chamber_of_deputies/classifiers/keras/test_model/']
        # Create the folder structure
        for directory in self.directories:
             if not os.path.exists(directory):
                 os.mkdir(directory)
        # Define the folder parameters for the train test
        self.train_data_dir = self.directories[1]
        self.validation_data_dir = self.directories[4]
        self.save_dir = self.directories[7]

        # Download the reimbursements
        for index, item in self.subject._X.iterrows():
            png_image = self.subject.download_doc(item.link)
            if png_image is not None:
                file_name = item.document_id+'.png'
                if index in [0, 2, 4, 5]:
                    # Save the suspicious in the positive training and validation folder
                    png_image.save(os.path.join(self.directories[2],
                    file_name),"PNG")
                    png_image.save(os.path.join(self.directories[5],
                    file_name),"PNG")
                else:
                    png_image.save(os.path.join(self.directories[3],
                    file_name),"PNG")
                    png_image.save(os.path.join(self.directories[6],
                    file_name),"PNG")


    def test_predict_true_when_generalized(self):
        assert_array_equal(np.repeat(True, 4),
                           self.prediction[[0, 2, 4, 5]])

    def test_predict_false_when_not_generalized(self):
        assert_array_equal(np.repeat(False, 2),
                          self.prediction[[1, 3]])
    def test_train(self):
        # It defines how many iterations will run to find the best model during traaining
        self.subject.epochs = 5
        # It influences the speed of your learning (Execution)
        self.subject.batch_size = 2
        self.subject.train(self.train_data_dir, self.validation_data_dir, self.save_dir)
        self.assertTrue(os.listdir(self.save_dir) != [])
        for directory in self.directories:
             if os.path.exists(directory):
                    shutil.rmtree(directory)
        self.assertFalse(os.path.exists(self.save_dir))
