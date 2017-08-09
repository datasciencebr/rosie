import os
import unicodedata
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image as pil_image
from sklearn.base import TransformerMixin
from wand.image import Image
from PIL import Image as pil_image
from io import BytesIO

class MealGeneralizationClassifier(TransformerMixin):
    """
    Meal Generalization Classifier.

    Dataset
    -------
    applicant_id : string column
        A personal identifier code for every person making expenses.

    category : category column
        Category of the expense. The model will be applied just in rows where
        the value is equal to "Meal".

    document_id : string column
        The identifier of the expense.

    year : string column
        The year the expense was generated.
    """

    COLUMNS = ['applicant_id', 'document_id', 'category', 'year']

    img_width, img_height = 300, 300

    def train(self, train_data_dir, validation_data_dir, save_dir):
        # Fix random seed for reproducibility
        seed = 2017
        np.random.seed(seed)

        nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
        nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

        print('no. of trained samples = ', nb_train_samples,
              ' no. of validation samples= ', nb_validation_samples)

        # Dimensions of our images.
        img_width, img_height = 300, 300

        # It defines how many iterations will run to find the best model
        epochs = 20
        # It influences the speed of your learning (Execution)
        batch_size = 15

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
        # Its a stack of 3 convolution layers with a ReLU activation followed by max-pooling layers
        # This is very similar to the architectures that Yann LeCun advocated in the 1990s
        # For image classification (with the exception of ReLU)
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # Convolutional network is a specific artificial neural network topology
        # Inspired by biological visual cortex and tailored for computer vision tasks.
        # Authour: Yann LeCun in early 1990s.
        # See http://deeplearning.net/tutorial/lenet.html for introduction.

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # This is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)
        # I put horizontal_flip as FALSE because we can not handwrite from right to left in Portuguese

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # This is the augmentation configuration we will use for testing:
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        # Generates more images for the validation step
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        # It allow us to save only the best model between the iterations
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(save_dir,"weights.hdf5"),
            verbose=1, save_best_only=True)

        # We set it as a parameter to save only the best model
        model.fit_generator(
            train_generator,
            callbacks=[checkpointer],
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

    def fit(self, X):
        # Load an existent Keras model
        self.keras_model = load_model(X)
        return self

    def transform(self, X=None):
        pass

    def predict(self, X):
        # Only use the import columns for our classifier
        self._X = X[self.COLUMNS]
        # Remove the reimbursements from categories different from Meal
        self._X = self._X[self.__applicable_rows(self._X)]
        # Creates a link to the chamber of deputies
        self._X = self.__document_url(self._X)
        # Assumes nothing is suspicious
        self._X['y'] = False
        result = []

        for index, item in self._X.iterrows():
            # Download the reimbursements
            png_image = self.__download_doc(item.link)
            if png_image is not None:
                x = img_to_array(png_image)
                x = np.expand_dims(x, axis=0)
                # Predict it in our model :D
                preds = self.keras_model.predict_classes(x, verbose=0)
                # Get the probability of prediciton
                prob = self.keras_model.predict_proba(x, verbose=0)
                # Keep the predictions with more than 80% of accuracy and the class 1 (suspicious)
                if(prob >= 0.8 and preds == 1):
                    result.append(True)
                else:
                    result.append(False)
            else:
                # Case the reimbursement can not be downloaded or convert it is classified as False
                result.append(False)

        self._X['y'] = result
        return self._X['y']

    def __applicable_rows(self, X):
        return (X['category'] == 'Meal')

    """ Creates a new column 'links' containing an url for the files in the chamber of deputies website
            Return updated Dataframe

        arguments:
        record -- Dataframe
    """

    def __document_url(self, X):
        X['link'] = ''
        links = list()
        for index, x in X.iterrows():
            base = "http://www.camara.gov.br/cota-parlamentar/documentos/publ"
            url = '{}/{}/{}/{}.pdf'.format(base, x.applicant_id, x.year, x.document_id)
            links.append(url)
        X['link'] = links
        return X

    """Download a pdf file and transform it to png
        Returns the png image using PIL image. It is necessary for Keras API

        arguments:
        url -- the url to chamber of deputies web site, e.g.,
        http://www.../documentos/publ/2437/2015/5645177.pdf

        Exception -- returns None
    """
    def __download_doc(self, url_link):
            try:
                # Open the resquest and get the file
                response = urlopen(url_link)
                # Default arguments to read the file and has a good resolution
                with Image(file=response, resolution=300) as img:
                    img.compression_quality = 99
                    # Chosen format to convert pdf to image
                    with img.convert('png') as converted:
                        # Converts the Wand image to PIL image
                        data = pil_image.open(BytesIO(converted.make_blob()))
                        data = data.convert('RGB')
                        hw_tuple = (self.img_height, self.img_width)
                        # Resizing of PIL image to fit our ML model
                        if data.size != hw_tuple:
                            data = data.resize(hw_tuple)
                        return data
            except Exception as ex:
                print("Error during pdf download")
                print(ex)
                # Case we get some exception we return None
                return None
