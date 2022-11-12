#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Common packages
import numpy as np
import pandas as pd
import warnings

# ML
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

#Keras/Tensorflow
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences as k_pad_sequences

# One hot encoding

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

oh_enc = OneHotEncoder()
le_enc = LabelEncoder()
oh_categories = {}
oh_class_indices = {}

def setup_onehot(df) :
    # Fit one hot encoder
    oh_enc.fit(class_columns(df))
    # Get categories
    oh_categories['class'] = oh_enc.categories_[0]    
    # Get indices
    oh_class_indices['class'] = np.arange(len(oh_categories['class']))

def setup_labelencoder(df) :
    # Fit one hot encoder
    le_enc.fit(class_columns(df))
    
def class_columns(df) :
    return np.asarray(df['class']).reshape(-1, 1) 

def label_encoding(df, class_name) :
    try:
        return le_enc.transform(class_columns(df))
    except:
        raise ValueError('Run setup_labelencoder first')
    
    

def onehot_encoding(df, class_name) :
    if oh_categories == {} :
        raise ValueError('Run setup_labelencoder first')
    
    return oh_enc.transform(class_columns(df)).toarray()[:,oh_class_indices[class_name]]

def read_data() :
    hdfs_train = pd.read_csv('./data/log_classification_train.csv')   
    hdfs_test_kaggle = pd.read_csv('./data/log_classification_test.csv')   
    return hdfs_train, hdfs_test_kaggle


def value_counts(data, attribute) :
    data[attribute].value_counts().plot(kind = 'bar')
    plt.ylabel('Count')
    plt.title(attribute)
    plt.show()

def load_sequences(seqs_df):
    seqs = []
    #Convert sequences in dataframe to list
    for seq in seqs_df['sequence'].to_list():
        clean_seq = replace(seq,[" ", "(", ")"],"").split(',')        
        clean_seq = [int(i)+1 for i in clean_seq]
        seqs.append(clean_seq)
    return seqs

def pad_sequences(seqs, max_len):
    #Use keras preprocessing to do padding
    padded_seqs = k_pad_sequences(seqs, maxlen=max_len, padding='pre',value=0)
    return padded_seqs

def replace(seq, symbols, new_symbol):
    for old_symbol in symbols:
        seq = seq.replace(old_symbol, new_symbol)
    return seq
    
def load_sequences_and_target(data, y_field_name = 'class', one_hot = True):    
    raw_sequences_X = load_sequences(data)    
    setup_onehot(data)
    setup_labelencoder(data)
    if one_hot:
        data_y = pd.DataFrame(onehot_encoding(data, y_field_name))    
    else:
        data_y = pd.DataFrame(label_encoding(data, y_field_name)) 

    return (raw_sequences_X, data_y)	

def train(model,
                train_X,
                train_y, 
                batch_size,
                epochs,
                validation_data_X,
				validation_data_y,                 
				patience,
				class_weights):				
	#Train
	##Callbacks    

	earlystopper = EarlyStopping(monitor='loss', patience=patience, verbose=1,restore_best_weights=True)
	
	training = model.fit(train_X
                        ,train_y
                        ,epochs=epochs
                        ,validation_data=(validation_data_X, validation_data_y)
                        ,batch_size=batch_size
                        ,callbacks=[earlystopper],
						class_weight = class_weights)
								
	return training, model

def split(data_X, data_y):
    """ 
    Split to train, test and validation. 
    
    @param data: Total dataset to split
    @return:  train data, validation data, test data
    """
    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=24)

    # Split train to train and validation datasets
    # Validation for use during learning
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train , test_size=0.1, random_state=24)

    return(X_train, X_test, X_val, y_train, y_test, y_val)

def eval_model(training, model, test_X, test_y, field_name = 'class'):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(training.history['accuracy'], label="Accuracy")
    ax[1].plot(training.history['val_accuracy'], label="Validation accuracy")
    ax[1].set_title('%s: accuracy' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Accuracy by category
    test_pred = model.predict(test_X)
    
    acc_by_category = np.logical_and((test_pred > 0.5), test_y).sum()/test_y.sum()
    acc_by_category.plot(kind='bar', title='Recall by %s' % field_name)
    plt.ylabel('Recall')
    plt.show()

    # Print metrics
    print("Classification report")
    test_pred = np.argmax(test_pred, axis=1)
    test_truth = np.argmax(test_y.values, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(metrics.classification_report(test_truth, test_pred, target_names=oh_categories[field_name]))

    # Loss function and accuracy
    test_res = model.evaluate(test_X, test_y.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])	
    
    
def predict_test(model, data):	
    prob = model.predict(data)
    pred = np.argmax(prob, axis=1).reshape(-1,1)    
    return pred

def gen_csv_file(test_ids, pred, class_name):
    output = np.stack((test_ids, pred), axis=-1)
    output = output.reshape([-1, 2])

    df = pd.DataFrame(output)
    df.columns = ['id','expected']
    
    df['expected'] = df['expected'].map(pd.Series(oh_categories[class_name]))    
    df.to_csv("kaggle_test_output.csv", index = False, index_label = False)
    return df

def load_test_sequences_and_generate_prediction_file(model, test_data, max_len):
    raw_sequences_X_test = load_sequences(test_data)
    padded_sequences = pad_sequences(raw_sequences_X_test, max_len)
    
    pred = predict_test(model, padded_sequences)
    
    test_ids = test_data['id']
    test_ids = np.array(test_ids).reshape(-1,1)

    return gen_csv_file(test_ids, pred, 'class')

  seqs_aug = seqs.copy()
def class_weights(df, class_name) :
    # http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    y = df[class_name]
    classes = np.unique(y)
    #weights = {i : 1 for i in range(categories[class_name].shape[0])}
    weights = compute_class_weight(class_weight = "balanced", classes = classes, y = y)
    class_weights = {k: v for k, v in enumerate(weights)}
    return class_weights

def build_model(units, vocab_size, embedding_size, max_len):
    optimizer = 'adam'
    loss = 'categorical_crossentropy'    

    model = Sequential()
    model.add(Embedding(vocab_size+1, embedding_size, input_length=max_len))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def build_improved_model(optimizer, loss, units, vocab_size, embedding_size, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size+1, embedding_size, input_length=max_len))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision()])
    return model

def grid_search(params, builder, cv):

    model = KerasClassifier(build_fn=builder)
    gs = GridSearchCV(estimator=model, param_grid=params, cv=cv, verbose=3, n_jobs=2)
    return gs

def sequences_augmentation (seqs, data_y, max_length, min_seq_len) :
  seqs_aug = seqs.copy()
  data_y_aug = data_y
  seqs_len = len(seqs)
  for i in range(seqs_len):
    if len(seqs[i]) > min_seq_len:
      seqs_aug.append(seqs[i][-max_length:])
      data_y_aug = data_y_aug.append(data_y[i:i+1])
  return (seqs_aug, data_y_aug)