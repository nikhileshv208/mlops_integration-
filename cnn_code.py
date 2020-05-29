from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


# In[ ]:


dataset=cifar100.load_data(label_mode='fine')


# In[ ]:


train,test=dataset
len(train)


# In[ ]:


X_train,y_train=train
X_train.shape


# In[ ]:


len(test)


# In[ ]:


X_test,y_test=test
X_test.shape


# In[ ]:


input_shape = (32,32,3)


# In[ ]:


X_train = X_train.astype('float32')
y_test = y_test.astype('float32')


# In[ ]:



X_train = X_train / 255
y_test = y_test / 255


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))





model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])





mind = model.fit(X_train, y_train,
            batch_size=50,
            epochs=100,
            verbose=1,
            validation_split=0.2)
