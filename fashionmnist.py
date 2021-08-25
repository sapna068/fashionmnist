
# coding: utf-8

# In[1]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam


# In[4]:


#imported csv files from uploaded csv files in jupyter note book
train= pd.read_csv(r'fashion-mnist_train.csv')
test= pd.read_csv(r'fashion-mnist_test.csv')


# In[5]:


train.describe() # no of rows and columns


# In[6]:


train.head()  #pixel data and each rows showing diff images  x will represent image data and y represnts the label


# In[8]:


#converting dataframes to numpy arrays
train_data = np.array(train, dtype='float32')   #tensor flow basically uses float 
test_data = np.array(test, dtype='float32')


# In[10]:


#slicing of tha arrays 
x_train = train_data[:, 1:] / 255  #take every single data from train_data ,slice from 1st column and will go till the end and rscale pixel data
y_train = train_data[:, 0]  #take all the rows and only 0th column

x_test = test_data[:, 1:] / 255   #255 is the maximum value, dividing by 255 expresses a 0-1 representation
y_test = test_data[:, 0]


# In[11]:


# spliting the training data into train and validate 

x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2,)                       #0.2 shows 20% is the validation data rest training data


# In[13]:


# to show how the image will be like
image = x_train[50, :].reshape((28, 28))  #taking 50 as row and all the columns in the row


# In[14]:


plt.imshow(image)
plt.show()


# In[15]:


image = x_train[53, :].reshape((28, 28))
plt.imshow(image)
plt.show()


# In[16]:


image = x_train[20, :].reshape((28, 28))
plt.imshow(image)
plt.show()


# In[18]:


image = x_train[43, :].reshape((28, 28))
plt.imshow(image)
plt.show()


# In[26]:


#building cnn model
im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))


# In[28]:


model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=im_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# In[29]:


model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[30]:


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))


# In[31]:


model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[37]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.fit(
    x_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(x_validate, y_validate),
)

