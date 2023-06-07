import numpy as np
from decision_tree import *
import pandas as pd

'''
In k-fold cross-validation, the original dataset is equally partitioned into k subparts or folds. Out of the k-folds or groups,
for each iteration, one group is selected as validation data, and the remaining (k-1) groups are selected as training data.
    @data (np.array of np.array of floats) it's all data 
    @attributes (np.array of strings) name tags of all data
    @k_fold (int) number of particions of data
    @t (int) type of generation algorithm for decision tree

    return mean of acuraccy score
'''
def k_fold_cross_validation(data, attributes, k_fold, t=0):
    # to split all in equal parts, we need to check if it's divisible by k_fold
    if data.shape[0] % k_fold != 0:
        # quit n rows to make it divisible
        data = np.delete(data, range(data.shape[0] % k_fold), axis=0)

    # split data by k_fold
    data = np.split(data, k_fold, axis=0)

    accuracy_score_total = np.array([])
    # we are gonna to iterate for every division we made to data, and this is gonna a be the k_fold
    for x, division in enumerate(data):
        # prepare the train data
        train = np.delete(data, x, axis=0)
        train = np.reshape(train, (division.shape[0] * (k_fold - 1), len(attributes)))
        X_train, y_train = train[:,:-1], train[:,-1]

        #prepare the test data
        X_test, y_test = division[:,:-1], division[:,-1]

        # create classificator from train data
        tree_act = decision_tree(X_train.T, y_train, attributes, t)

        # made the prediccions for X_test data
        y_pred = tree_act.predict(X_test, attributes)

        # we check the acuraccy of the prediccions y_pred
        accuracy_score_total = np.append(accuracy_score_total, acuraccy_score(y_test, y_pred))

    return np.mean(accuracy_score_total)

'''
Leave-one-out cross-validation is an exhaustive cross-validation technique. It is a category of k-fold with the case of k=len(data)
    @data (np.array of np.array of floats) it's all data 
    @attributes (np.array of strings) name tags of all data
    @t (int) type of generation algorithm for decision tree

    return mean of acuraccy score
'''       
def leave_one_out(data, attributes, t=0):
    k_fold = len(data)
    return k_fold_cross_validation(data, attributes, k_fold, t=t)

'''
The holdout technique is an exhaustive cross-validation method, that randomly splits the dataset into train and test data depending on data analysis.
    @data (np.array of np.array of floats) it's all data 
    @attributes (np.array of strings) name tags of all data
    @k_fold (int) number of particions of data
    @t (int) type of generation algorithm for decision tree

    return mean of acuraccy score
'''
def holdout_cross_validation(data, attributes, k_fold, t=0):
    accuracy_score_total = np.array([])

    # we are gonna to iterate for every division we made to data, and this is gonna a be the k_fold
    for k in range(k_fold):
        # create random divisions
        rand = np.random.choice([False, True], len(data), p=[0.70, 0.30])

        # prepare the train data
        train = data[np.invert(rand)]
        X_train, y_train = train[:,:-1], train[:,-1]

        #prepare the test data
        test = data[rand]
        X_test, y_test = test[:,:-1], test[:,-1]

        # create classificator from train data
        tree_act = decision_tree(X_train.T, y_train, attributes, t)

        # made the prediccions for X_test data
        y_pred = tree_act.predict(X_test, attributes)

        # we check the acuraccy of the prediccions y_pred
        accuracy_score_total = np.append(accuracy_score_total, acuraccy_score(y_test, y_pred))

    return np.mean(accuracy_score_total)

'''
Given the predictions and the ground truth, returns the confusion matrix of the predictions by comparing it to the ground truth
    @y_test (np.array) correct y from the data
    @y_train (np.array) predicted y from the data
    @num_classes (int) number of classes to categorize y

    return confussion_matrix (np.array of np.array) 
'''
def confusion_matrix(y_test, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(len(y_pred)):
        confusion_matrix[int(y_pred[i]), int(y_test[i])] += 1

    return confusion_matrix

'''
Returns which % of the predictions match with the ground truth
    @y_test (np.array) correct y from the data
    @y_train (np.array) predicted y from the data

    return accuracy score (all matches) / all * 100
'''
def acuraccy_score(y_test, y_pred):
    return np.count_nonzero((y_pred == y_test) == True) / len(y_test) * 100




'''
Apply the algorithm of random forest
    @dataset (dataframe) data
    @attributes (np.array) atributtes of dataset
    @times (int) number of trees that you will create

    return y_pred(list),test_data(ndarray),class_labels(ndarray)
'''

def random_forest(dataset,attributes,times):


    if np.sqrt(dataset.shape[1])-int(np.sqrt(dataset.shape[1])) >= 0.5:
        num_attributes= int(np.sqrt(dataset.shape[1]))+1
    else:
        num_attributes= int(np.sqrt(dataset.shape[1]))
    n=0

    #escogemos que data estara en train y cual en test
    train_indices=np.random.choice(np.arange(dataset.shape[0]),100,replace=True)
    train=dataset.to_numpy()[train_indices]
    test=np.delete(dataset.to_numpy(),np.unique(train_indices),axis=0)

    y_pred=np.empty((times,test.shape[0]))



    # reemplaza todos los nans del train con el valor más frecuente en la fila
    for i in range(train.shape[1]):
        uniques, counts = np.unique(train[:, i][~np.isnan(train[:, i])], return_counts=True)
        train[:, i][np.isnan(train[:, i])] = uniques[np.where(counts == np.amax(counts))][0]

    #tratamos los continuos
    train[:, :-1] = contineous_treatment(train.T)
    test[:, :-1] = contineous_treatment(test.T)
    
    while(n<times):
        train_data, test_data =pd.DataFrame(train), pd.DataFrame(test)
        train_data.columns, test_data.columns = attributes,attributes        
        #escogemos al azar (num_attributes) atributos de attributes - la ultima columna                     
        #guardamos en dataTrain solo las columnas atributos escogidas
        train_data_atributes=np.append(np.random.choice(attributes[:-1],num_attributes,replace=False),'num')
        train_data=train_data[train_data_atributes]     

        train_data, test_data = train_data.to_numpy(), test_data.to_numpy()

        train_X = train_data[:, :-1]
        train_y = train_data[:, -1]

        test_X = test_data[:, :-1]
        test_y = test_data[:, -1]

        t=0
        tree = decision_tree(train_X.T, train_y.T, train_data_atributes, t)
        tree.viz("B")            

        y_pred[n] = tree.predict(test_X, attributes)        
        n=n+1
    
    valores=[]
    conteos=[]

    #vemos cual clase es majoritaria de cada dato
    for col in np.arange(y_pred.shape[1]):
        valores_aux,conteos_aux=np.unique(y_pred[:,col],return_counts=True)
        valores.append(valores_aux)
        conteos.append(conteos_aux)
    argMaximo=[np.argmax(row) for row in conteos]
    
    y_pred=[valores[row][argMaximo[row]] for row in range(len(valores))]
    class_labels=np.unique(train_y)

    return y_pred,test_data,class_labels
