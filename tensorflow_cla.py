import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

batch_size =32
img_height = 200
img_width = 200
class_names=[]
num_classes = 2
data_dir=r'D:\PycharmProjects\pythonProject\jiangnan2020\ten'

def load_resource():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    print(type(train_ds))
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    global class_names
    class_names = train_ds.class_names
    print(class_names)
    global num_classes
    num_classes=len(class_names)

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #   for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title(class_names[labels[i]])
    #     plt.axis("off")
    # plt.show()

    # AUTOTUNE = tf.data.experimental.AUTOTUNE

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds,val_ds

def rescaling(train_ds):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

def train1(train_ds,val_ds):

    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs=15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    model.save('my_model1.h5')
    model = tf.keras.models.load_model('my_model1.h5')
    model.summary()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return model

def transform(train_ds):
    def augment(image):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image

    # plt.figure(figsize=(10, 10))
    normalized_ds =(train_ds .map(lambda x,y:(augment(x),y)))
    # for images, labels in train_ds:
    #     for i in range(0, 9):
    #         images=augment(images)
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[0].numpy().astype('uint8'))
    #         plt.axis('off')
    # plt.show()

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = normalized_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return train_ds

def train(train_ds,val_ds):
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )

    model = Sequential([
      data_augmentation,
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs =20
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    model.save('my_modelone111.h5')
    # new_model = tf.keras.models.load_model('my_modelone.h5')
    # new_model.summary()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return model

def get_predict(model):
    test_images_path=r'D:\PycharmProjects\pythonProject\jiangnan2020\test\test'
    row_list=[]
    for file in os.listdir(test_images_path):
        test_path=os.path.join(test_images_path,file)

        img = keras.preprocessing.image.load_img(
            test_path, target_size=(img_height, img_width)
        )

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )

        label=np.argmax(score)
        row=[]
        id=re.sub('.jpg','',file)
        row.append(id)
        row.append(label)
        row_list.append(row)
    with open('tenone111.csv', 'w', encoding='utf-8', newline='') as f:
        f_reader=csv.writer(f)
        f_reader.writerow(['id','label'])
        f_reader.writerows(row_list)

if  __name__=='__main__':
    train_ds,val_ds=load_resource()
    train_ds1=transform(train_ds)

    model=train(train_ds,val_ds)
    history=model.fit(
        train_ds1,
        validation_data=val_ds,
        epochs=20
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(20)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # get_predict(model)
    model.save('mymodelone111.h5')
