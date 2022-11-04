import os
from os.path import exists
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
from sklearn.model_selection import cross_val_score, KFold

import pickle
import gc
import numpy as np
import time
import pandas as pd
import datatable as dt
import matplotlib.pyplot as plt
import gc

### global variable
activation_func = 'relu' # for all middle layers
activation_func2 = 'linear' # for output layer to output unbounded gene-effect scores
init = 'he_uniform'
dense_layer_dim = 250


def load_data(filename):
    df_dt = dt.fread(filename, sep="\t")
    gene_names = df_dt[0].to_list()[0]
    sample_names = list(df_dt.names[1:])
    data_np = df_dt[:, 1:].to_numpy()
    data_np = np.transpose(data_np)
    gc.collect()
    
    return data_np, sample_names, gene_names

def load_data_prediction(filename):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names


def trainset_load(npz_path="prediction/data/ccl_complete_data_501CCL_1298DepOI_614727samples.npz", 
                  filename="training_custom"):
    
    if exists(npz_path) is False:
        # expression
        if exists(TEMP_PATH + filename + "_exp_training.npy") is False:
            data_exp, sample_names_exp, property_names_exp = load_data(TRAIN_PATH + filename + "_exp_training.txt")
            np.save(TEMP_PATH + filename + "_exp_training.npy", data_exp)
        else:
            data_exp = np.load(TEMP_PATH + filename + "_exp_training.npy")
        print("exp load.. completed")    

        # mutation
        if exists(TEMP_PATH + filename + "_mut_training.npy") is False:
            data_mut, sample_names_mut, property_names_mut = load_data(TRAIN_PATH + filename + "_mut_training.txt")
            np.save(TEMP_PATH + filename + "_mut_training.npy", data_mut)
        else:
            data_mut = np.load(TEMP_PATH + filename + "_mut_training.npy")
        print("mut load.. completed")

        # copy number alteration
        if exists(TEMP_PATH + filename + "_cna_training.npy") is False:
            data_cna, sample_names_cna, property_names_cna = load_data(TRAIN_PATH + filename + "_cna_training.txt")
            np.save(TEMP_PATH + filename + "_cna_training.npy", data_cna)
        else:
            data_cna = np.load(TEMP_PATH + filename + "_cna_training.npy")
        print("cna load.. completed")

        # methylation
        if exists(TEMP_PATH + filename + "_meth_training.npy") is False:
            data_meth, sample_names_meth, property_names_meth = load_data(TRAIN_PATH + filename + "_meth_training.txt")
            np.save(TEMP_PATH + filename + "_meth_training.npy", data_meth)
        else:
            data_meth = np.load(TEMP_PATH + filename + "_meth_training.npy")
        print("meth load.. completed")

        # dependency score
        if exists(TEMP_PATH + filename + "_DepScore_training.npy") is False:
            data_dep, sample_names_dep, property_names_dep = load_data(TRAIN_PATH + filename + "_DepScore_training.txt")
            np.save(TEMP_PATH + filename + "_DepScore_training.npy", data_dep)
        else:
            data_dep = np.load(TEMP_PATH + filename + "_DepScore_training.npy")
        print("dep load.. completed")

        # fingerprint
        if exists(TEMP_PATH + filename + "_fingerprint_training.npy") is False:
            data_fprint, sample_names_fprint, property_names_fprint = load_data(TRAIN_PATH + filename + "_fingerprint_training.txt")
            np.save(TEMP_PATH + filename + "_fingerprint_training.npy", data_fprint)
        else:
            data_fprint = np.load(TEMP_PATH + filename + "_fingerprint_training.npy")
        print("fingerprint load.. completed")

        np.savez_compressed(npz_path, 
                            data_exp=data_exp, data_mut=data_mut, data_cna=data_cna, data_meth=data_meth,
                            data_dep=data_dep, data_fprint=data_fprint)
    else :
        data = np.load(npz_path)
        data_exp = data['data_exp']
        data_mut = data['data_mut']
        data_cna = data['data_cna']
        data_meth = data['data_meth']
        data_fprint = data['data_fprint']
        data_dep = data['data_dep']
    
    # gc load
    gc.collect()
    
    return data_exp, data_mut, data_cna, data_meth, data_fprint, data_dep


def full_model(premodel_exp, premodel_mut, premodel_meth, premodel_cna, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_mut = Sequential()
        model_mut.add(Dense(1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                            weights=premodel_mut[0], trainable=True))
        model_mut.add(Dense(100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                            trainable=True))
        model_mut.add(Dense(50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                            trainable=True))

        model_exp = Sequential()
        model_exp.add(Dense(500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                            weights=premodel_exp[0], trainable=True))
        model_exp.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                            trainable=True))
        model_exp.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                            trainable=True))

        model_cna = Sequential()
        model_cna.add(Dense(500, input_dim=premodel_cna[0][0].shape[0], activation=activation_func,
                            weights=premodel_cna[0], trainable=True))
        model_cna.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_cna[1],
                            trainable=True))
        model_cna.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_cna[2],
                            trainable=True))

        model_meth = Sequential()
        model_meth.add(Dense(500, input_dim=premodel_meth[0][0].shape[0], activation=activation_func,
                             weights=premodel_meth[0], trainable=True))
        model_meth.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_meth[1],
                             trainable=True))
        model_meth.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_meth[2],
                             trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init,
                             trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_mut.output, model_exp.output, 
                              model_cna.output, model_meth.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=250, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_mut.input, model_exp.input, model_cna.input, model_meth.input, model_gene.input], model_pre)
        
        return model_final

def exp_mut_cna_model(premodel_exp, premodel_mut, premodel_cna, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_mut = Sequential()
        model_mut.add(Dense(1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                            weights=premodel_mut[0], trainable=True))
        model_mut.add(Dense(100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                            trainable=True))
        model_mut.add(Dense(50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                            trainable=True))

        model_exp = Sequential()
        model_exp.add(Dense(500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                            weights=premodel_exp[0], trainable=True))
        model_exp.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                            trainable=True))
        model_exp.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                            trainable=True))

        model_cna = Sequential()
        model_cna.add(Dense(500, input_dim=premodel_cna[0][0].shape[0], activation=activation_func,
                            weights=premodel_cna[0], trainable=True))
        model_cna.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_cna[1],
                            trainable=True))
        model_cna.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_cna[2],
                            trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        # model_gene.add(Dense(1000, input_dim=data_fprint.shape[1], activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_mut.output, model_exp.output, 
                              model_cna.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=200, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_mut.input, model_exp.input, model_cna.input, model_gene.input], model_pre)
        
        return model_final
    
    
def exp_mut_model(premodel_exp, premodel_mut, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_mut = Sequential()
        model_mut.add(Dense(1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                            weights=premodel_mut[0], trainable=True))
        model_mut.add(Dense(100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                            trainable=True))
        model_mut.add(Dense(50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                            trainable=True))

        model_exp = Sequential()
        model_exp.add(Dense(500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                            weights=premodel_exp[0], trainable=True))
        model_exp.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                            trainable=True))
        model_exp.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                            trainable=True))
        
        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init,
                             trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_mut.output, model_exp.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=150, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_mut.input, model_exp.input, model_gene.input], model_pre)
        
        return model_final    

def exp_model(premodel_exp,data_fprint_shape):
    with tf.device('/cpu:0'):
        model_exp = Sequential()
        model_exp.add(Dense(500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                            weights=premodel_exp[0], trainable=True))
        model_exp.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                            trainable=True))
        model_exp.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                            trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_exp.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=100, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_exp.input, model_gene.input], model_pre)
        
        return model_final      
    
def mut_model(premodel_mut, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_mut = Sequential()
        model_mut.add(Dense(1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                            weights=premodel_mut[0], trainable=True))
        model_mut.add(Dense(100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                            trainable=True))
        model_mut.add(Dense(50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                            trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init,
                             trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_mut.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=100, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_mut.input, model_gene.input], model_pre)
        
        return model_final

def meth_model(premodel_meth, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_meth = Sequential()
        model_meth.add(Dense(500, input_dim=premodel_meth[0][0].shape[0], activation=activation_func,
                             weights=premodel_meth[0], trainable=True))
        model_meth.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_meth[1],
                             trainable=True))
        model_meth.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_meth[2],
                             trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init,
                             trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_meth.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=200, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_meth.input, model_gene.input], model_pre)
        
        return model_final
    
    
def cna_model(premodel_cna, data_fprint_shape):
    with tf.device('/cpu:0'):
        model_cna = Sequential()
        model_cna.add(Dense(500, input_dim=premodel_cna[0][0].shape[0], activation=activation_func,
                            weights=premodel_cna[0], trainable=True))
        model_cna.add(Dense(200, input_dim=500, activation=activation_func, weights=premodel_cna[1],
                            trainable=True))
        model_cna.add(Dense(50, input_dim=200, activation=activation_func, weights=premodel_cna[2],
                            trainable=True))

        # subnetwork of gene fingerprints
        model_gene = Sequential()
        model_gene.add(Dense(1000, input_dim=data_fprint_shape, activation=activation_func, kernel_initializer=init,
                             trainable=True))
        model_gene.add(Dense(100, input_dim=1000, activation=activation_func, kernel_initializer=init, trainable=True))
        model_gene.add(Dense(50, input_dim=100, activation=activation_func, kernel_initializer=init, trainable=True))

        conc = Concatenate()([model_cna.output, model_gene.output])

        model_pre = Dense(dense_layer_dim, input_dim=100, activation=activation_func, kernel_initializer=init,
                              trainable=True)(conc)
        model_pre = Dense(dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, kernel_initializer=init,
                              trainable=True)(model_pre)
        model_pre = Dense(1, input_dim=dense_layer_dim, activation=activation_func2, kernel_initializer=init,
                              trainable=True)(model_pre)

        model_final = Model([model_cna.input, model_gene.input], model_pre)
        
        return model_final    


class generator(tf.keras.utils.Sequence):
    def __init__(self, x_mut, x_exp, x_cna, x_meth, x_fprint, y_dep, batch_size):
        self.x_mut, self.x_exp, self.x_cna, self.x_meth, self.x_fprint, self.y_dep = x_mut, x_exp, x_cna, x_meth, x_fprint, y_dep
        self.bs = batch_size
    
    def __len__(self):
        return (len(self.y_dep) - 1) // self.bs + 1
    
    def __getitem__(self,idx):
        start, end = idx * self.bs, (idx+1) * self.bs
        return (self.x_mut[start:end], self.x_exp[start:end], self.x_cna[start:end], self.x_meth[start:end], self.x_fprint[start:end]), self.y_dep[start:end]

class generator3(generator):
    def __init__(self, x_mut, x_exp, x_cna, x_fprint, y_dep, batch_size):
        self.x_mut, self.x_exp, self.x_cna, self.x_fprint, self.y_dep = x_mut, x_exp, x_cna, x_fprint, y_dep
        self.bs = batch_size
        
    def __getitem__(self, idx):
        start, end = idx * self.bs, (idx+1) * self.bs
        return (self.x_mut[start:end], self.x_exp[start:end], self.x_cna[start:end], self.x_fprint[start:end]), self.y_dep[start:end]
    
class generator2(generator):
    def __init__(self, x_mut, x_exp, x_fprint, y_dep, batch_size):
        self.x_mut, self.x_exp, self.x_fprint, self.y_dep = x_mut, x_exp, x_fprint, y_dep
        self.bs = batch_size
        
    def __getitem__(self, idx):
        start, end = idx * self.bs, (idx+1) * self.bs
        return (self.x_mut[start:end], self.x_exp[start:end], self.x_fprint[start:end]), self.y_dep[start:end]

class generator1(generator):
    def __init__(self, x_mut, x_fprint, y_dep, batch_size):
        self.x_mut, self.x_fprint, self.y_dep = x_mut, x_fprint, y_dep
        self.bs = batch_size
        
    def __getitem__(self, idx):
        start, end = idx * self.bs, (idx+1) * self.bs
        return (self.x_mut[start:end], self.x_fprint[start:end]), self.y_dep[start:end]
    
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def model_train_vis(history):
    coeff = history.history['coeff_determination']
    val_coeff = history.history['val_coeff_determination']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(coeff))
    
    plt.plot(epochs, coeff, 'bo', label='Training')
    plt.plot(epochs, val_coeff, 'b', label='Validation')
    plt.title('Training and validation Coeff determination(R-squared)')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training')
    plt.plot(epochs, val_loss, 'b', label='Validation')
    plt.title('Training and validation loss(MSE)')
    plt.legend()

    plt.show()
