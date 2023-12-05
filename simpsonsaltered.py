import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

print(f'tensorflow version: {tf.__version__}')

PATH_DATASET = '/Users/caiolima/Documents/inteligência artificial/lista extra 2/archive/dataset_personagens/dataset_personagens'
print('List dir:')
for file in os.listdir(PATH_DATASET):
  print(file)

train_dir = os.path.join(PATH_DATASET, 'training_set')
validation_dir = os.path.join(PATH_DATASET, 'test_set')

train_bart_dir = os.path.join(train_dir, 'bart')
train_homer_dir = os.path.join(train_dir, 'homer')
validation_bart_dir = os.path.join(validation_dir, 'bart')
validation_homer_dir = os.path.join(validation_dir, 'homer')

num_bart_tr = len(os.listdir(train_bart_dir))
num_homer_tr = len(os.listdir(train_homer_dir))

num_bart_val = len(os.listdir(validation_bart_dir))
num_homer_val = len(os.listdir(validation_homer_dir))

total_train = num_bart_tr + num_homer_tr
total_val = num_bart_val + num_homer_val

print('total training bart images:', num_bart_tr)
print('total training homer images:', num_homer_tr)

print('total validation bart images:', num_bart_val)
print('total validation homer images:', num_homer_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

img_bart = os.path.join(train_bart_dir, os.listdir(train_bart_dir)[5])
PIL.Image.open(img_bart)

img_homer = os.path.join(train_homer_dir, os.listdir(train_homer_dir)[13])
PIL.Image.open(img_homer)

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# dados treino com aumento do conjunto de imagems
train_image_gen = ImageDataGenerator(rescale = 1./255,
                               rotation_range = 7,
                               horizontal_flip = True,
                               shear_range = 0.2,
                               height_shift_range = 0.05,
                               zoom_range = 0.2)

val_image_gen = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=IMG_SIZE,
                                                     class_mode='binary')

val_data_gen = val_image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 shuffle=True,
                                                 target_size=IMG_SIZE,
                                                 class_mode='binary')

sample_training_images, _ = next(val_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(sample_training_images[:5])
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.>>', end='')

steps_per_epoch = train_data_gen.samples // train_data_gen.batch_size
validation_steps = val_data_gen.samples // val_data_gen.batch_size
epochs=100
    
history = model.fit(
    train_data_gen,
    epochs=epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data_gen,
    validation_steps=validation_steps,
    callbacks=[PrintDot()],
    verbose=0
    )

# Dataframe results model
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.describe()

def subplots(df, vline=None):
  cols_names = df.columns.tolist()
  cases = list(range(len(cols_names[:-1])))
  plot_params = {
      'axes.titlesize': 12,
      'xtick.labelsize': 9,
      'ytick.labelsize': 9,
      }
  with plt.rc_context(plot_params):
    with plt.style.context('seaborn-darkgrid'):
      fig, axs = plt.subplots(2, 2, figsize=(8, 4), constrained_layout=True, sharex=True)
      for ax, i in zip(axs.flat, cases):
          ax.set_title(cols_names[:-1][i])
          ax.plot(df['epoch'],  df[cols_names[:-1][i]])
          #vline = ax.axvline(x=2, color='#7fb800')
          x = ax.axvline(x=vline, color='#ffb400') if vline != None else False
      fig.text(0.5, -0.05, 'epoch', ha='center')

subplots(hist)

eval_results = model.evaluate(val_data_gen)
print('Testing set Accuracy: {:.2f}'.format(eval_results[1]))
print('Testing set Accuracy: {:2.2%}'.format(eval_results[1]))

test_bart = os.path.join(train_bart_dir, os.listdir(train_bart_dir)[5])
homer_test = os.path.join(train_homer_dir, os.listdir(train_homer_dir)[13])
inv_map = {train_data_gen.class_indices[k] : k for k in train_data_gen.class_indices}
inv_map

imagem_teste = image.load_img(homer_test,
                              target_size = (160,160))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

previsao = model.predict(imagem_teste).flatten()
prev_name = tf.where(previsao < 0.5, 0, 1).numpy()

inv_map[prev_name[0]], previsao

img1, nome = next(val_data_gen)
pred = model.predict(img1).flatten()
pred = tf.where(pred < 0.5, 0, 1)
plt.imshow(img1[0])
title = inv_map[pred.numpy()[0]]
plt.title(f'Predict name: {title}')
plt.axis("off")
plt.show()

