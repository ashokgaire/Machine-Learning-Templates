################## Convolution Neural Network ############################

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers impoer Dense

# Initialising the cnn
classifier = Sequential()

#setp 1 : convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64, 64 ,3), activation= 'relu'))

#step 2: pooling
classifier.add(MaxPooling2D( pool_size = (2,2)))

#setp 3: flattening
classifier.add(Flatten())

#step 4: full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(unit =1 , activation ='sigmoid'))

# compling the cnn
classifier.compile(optimizer = 'adam', loss= 'binary_crossentrophy', metrics = 'accuracy')


# fitting the cnn to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

traning_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                calss_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                calss_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)