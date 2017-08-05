import os
import unicodedata
import numpy as np
import pandas as pd
import urllib
from sklearn.base import TransformerMixin
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import img_to_array
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

    COLS = ['applicant_id',
            'document_id',
            'category',
            'year']


    img_width, img_height = 300, 300

    def train(self,train_data_dir,validation_data_dir,save_dir):
        #fix random seed for reproducibility
        seed = 2017
        np.random.seed(seed)

        nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
        nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

        print('no. of trained samples = ', nb_train_samples, ' no. of validation samples= ',nb_validation_samples)

        #dimensions of our images.
        img_width, img_height = 300, 300

        epochs = 20
        batch_size = 15

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
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

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        #this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)#As you can see i put it as FALSE and on link example it is TRUE
        #Explanation, there no possibility to write in a reverse way :P

        #this is the augmentation configuration we will use for testing:
        #only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        #It allow us to save only the best model between the iterations
        checkpointer = ModelCheckpoint(filepath=save_dir+"weights.hdf5", verbose=1, save_best_only=True)

        model.fit_generator(
            train_generator,
             callbacks=[checkpointer], #And we set the parameter to save only the best model
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

    def fit(self, X):
        #Load an existent Keras model
        self.keras_model = load_model(X)
        return self

    def transform(self, X=None):
        pass

    def predict(self, X):
        self._X = X[self.COLS]
        self._X = self._X[self.__applicable_rows(self._X)]
        self._X = self.__document_url(self._X)
        self._X['y']=False
        result=[]

        for index, item in self._X.iterrows():

            png_image = self.__download_doc(item.link)
            if png_image is not None :
                x = img_to_array(png_image)
                x = np.expand_dims(x, axis=0)

                preds = self.keras_model.predict_classes(x, verbose=0) #predict it in our model :D
                prob = self.keras_model.predict_proba(x, verbose=0) #get the probability of prediciton
                if(prob>=0.8 and preds==1):#Only keep the predictions with more than 80% of accuracy and the class 1 (suspicious)
                    result.append(True)
                else:
                    result.append(False)
            else:
                result.append(False)

        self._X['y']=result
        return self._X['y']

    def __applicable_rows(self, X):
        return (X['category'] == 'Meal')


    """convert the row of a dataframe to a string represinting the url for the files in the chamber of deputies
        Return a string to access the files in the chamber of deputies web site

        arguments:
        record -- row of a dataframe
    """

    def __document_url(self,X):
        X['link']=''
        links=list()
        for index, x in X.iterrows():
            links.append('http://www.camara.gov.br/cota-parlamentar/documentos/publ/{}/{}/{}.pdf'.format(x.applicant_id,x.year, x.document_id))
        X['link']=links
        return X

    """Download a pdf file and transform it to png
        Returns the png image using PIL image

        arguments:
        url -- the pdf url to chamber of deputies web site, e.g., http://www.../documentos/publ/2437/2015/5645177.pdf

        Exception -- returns None
    """
    def __download_doc(self,url_link):
            #using the doc id as file name
            try:
                #open the resquest and get the file
                with urllib.request.urlopen(url_link) as response:
                    # return the name of the pdf converted to png
                    #Default arguments to read the file and has a good resolution
                    with Image(file=response, resolution=300) as img:
                        img.compression_quality = 99
                        #Format choosed to convert the pdf to image
                        with img.convert('png') as converted:
                            print(converted)
                            data = pil_image.open(BytesIO(converted.make_blob()))
                            data = data.convert('RGB')
                            hw_tuple = (self.img_height, self.img_width)
                            if data.size != hw_tuple:
                                 data = data.resize(hw_tuple)
                            print(data)
                            return data
            except Exception as ex:
                print("Error during pdf download")
                print(ex)
                return None #case we get some exception we return None
