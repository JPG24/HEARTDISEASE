import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from decision_tree import *
from testing import *
'''
age -> age in years                                                                                 (continues)
sex -> female 0, male 1                                                                             (categorical)
cp -> chest pain type: typical angina 1, atypical angina 2, non-anginal pain 3, asymptomatic 4      (categorical)
trestbps -> resting blood pressure (in mm Hg on admission to the hospital)                          (continues)
chol -> serum cholestoral in mg/dl                                                                  (continues)
fbs -> (fasting blood sugar > 120 mg/dl)  (true 1; false 0)                                         (categorical)
restecg ->  resting electrocardiographic results (normal 0, having ST-T wave abnormality 1          (categorical)
            showing probable or definite left ventricular hypertrophy by Estes' criteria 2
thalach -> maximum heart rate achieved                                                              (continues)
exang -> exercise induced angina (yes 1, no 0)                                                      (categorical)
oldpeak -> ST depression induced by exercise relative to rest                                       (continues)
slope -> he slope of the peak exercise ST segment (upsloping 1, flat 2, downsloping 3)              (categorical)
ca -> number of major vessels (0-3) colored by flourosopy                                           (categorical)
thal -> 3 = normal, 6 = fixed defect, 7 = reversable defect                                         (categorical)
num -> diagnosis of heart disease < 50% diameter narrowing 0, > 50% diameter narrowing 1            (categorical)
'''

def load_data(path): 
    hospital_va = pd.read_csv(f"{path}/processed.va.data", header=None)
    hospital_cleveland = pd.read_csv(f"{path}/processed.cleveland.data", header=None)
    hospital_hungarian = pd.read_csv(f"{path}/processed.hungarian.data", header=None)
    hospital_switzerland = pd.read_csv(f"{path}/processed.switzerland.data", header=None)

    hospitals = pd.concat([hospital_va, hospital_hungarian, hospital_cleveland, hospital_switzerland])

    # Headers for atributes
    header = {0 : "age", 1 : "sex", 2 : "cp", 3 : "trestbps", 4 : "chol", 5 : "fbs", 6 : "restecg",  \
                7 : "thalach", 8 : "exang", 9 : "oldpeak", 10 : "slope", 11 : "ca", 12 : "thal", 13 : "num"}

    hospitals = hospitals.rename(columns=header)

    # Replace all '?' by NaN because it is the sign to express that it misses values
    # Convert all values to float
    for h in header.values():
        hospitals = hospitals.replace({h : {'?' : np.nan}})
        hospitals[h] = hospitals[h].values.astype(np.float64)

    return hospitals

def split_train_test(data, p):
    n_test = int(data.shape[0] * p)
    print("n_test: ", n_test)
    np.random.shuffle(data)
    return data[n_test:,:], data[:n_test,:]


def C(dataset, t, k_fold):
    #Delete "slope", "ca", "thal" becasue as many NaN values
    dataset = dataset.drop(["slope", "ca", "thal"], axis=1)

    # We temporarily drop NaN until we decide what to do with them
    dataset = dataset.dropna()

    print(f"New shape of the dataset: {dataset.shape[0]} rows and {dataset.shape[1]} columns")

    # Discretaze continues values of dataset
    dataset["age"] = pd.qcut(dataset["age"], 4, labels=[0, 1, 2, 4])
    dataset["trestbps"] = pd.qcut(dataset["trestbps"], 3, labels=[0, 1, 2]) 
    dataset["chol"] = pd.qcut(dataset["chol"], 3, labels=[0, 1, 2])
    dataset["thalach"] = pd.qcut(dataset["thalach"], 3, labels=[0, 1, 2])
    dataset["oldpeak"] = pd.qcut(dataset["oldpeak"], 3, labels=[0, 1, 2])

    atribute_labels = np.array(list(dataset))

    # Convert pd.DataFrame to np.array
    data = dataset.values

    train_data, test_data = split_train_test(data, 0.3)

    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]

    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    class_labels = np.unique(train_y)

    print(dataset.describe())

    tree = decision_tree(train_X.T, train_y.T, atribute_labels, t)
    tree.viz("C")
    
    y_pred = tree.predict(test_X, atribute_labels)
    print(confusion_matrix(test_data[:, -1], y_pred, len(class_labels)))
    print(f"{acuraccy_score(test_data[:, -1], y_pred)}% of accuracy")

    all_labels = count_possible_labels(dataset.T)

    print("Holdout Cross Validation: ", holdout_cross_validation(data, atribute_labels, k_fold, t))
    print("Cross Validation: ", k_fold_cross_validation(data, atribute_labels, k_fold, t))
    #print("Leave One Out: ", leave_one_out(data, atribute_labels, t))

def B(dataset, t, k_fold):
    # Discretaze continues values of dataset
    dataset["age"] = pd.qcut(dataset["age"], 4, labels=[0, 1, 2, 4])
    dataset["trestbps"] = pd.qcut(dataset["trestbps"], 3, labels=[0, 1, 2]) 
    dataset["chol"] = pd.qcut(dataset["chol"], 3, labels=[0, 1, 2])
    dataset["thalach"] = pd.qcut(dataset["thalach"], 3, labels=[0, 1, 2])
    dataset["oldpeak"] = pd.qcut(dataset["oldpeak"], 3, labels=[0, 1, 2])

    atribute_labels = np.array(list(dataset))

    # Convert pd.DataFrame to np.array
    data = dataset.values

    train_data, test_data = split_train_test(data, 0.3)

    # replace all nans from train with the most frequent value in the row
    for i in range(train_data.shape[1]):
        uniques, counts = np.unique(train_data[:, i][~np.isnan(train_data[:, i])], return_counts=True)
        train_data[:, i][np.isnan(train_data[:, i])] = uniques[np.where(counts == np.amax(counts))][0]

    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]

    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    class_labels = np.unique(train_y)

    t = 0

    tree = decision_tree(train_X.T, train_y.T, atribute_labels, t)
    tree.viz("B")
    
    y_pred = tree.predict(test_X, atribute_labels)
    print(confusion_matrix(test_data[:, -1], y_pred, len(class_labels)))
    print(f"{acuraccy_score(test_data[:, -1], y_pred)}% of accuracy")

    all_labels = count_possible_labels(dataset.T)
    k_fold = 3

    print("Holdout Cross Validation: ", holdout_cross_validation(data, atribute_labels, k_fold, t))
    print("Cross Validation: ", k_fold_cross_validation(data, atribute_labels, k_fold, t))
    #print("Leave One Out: ", leave_one_out(data, atribute_labels, t))

def A(dataset, t, k_fold):
    atribute_labels = np.array(list(dataset))

    # Convert pd.DataFrame to np.array
    data = dataset.values

    train_data, test_data = split_train_test(data, 0.3)

    # replace all nans from train with the most frequent value in the row
    for i in range(train_data.shape[1]):
        uniques, counts = np.unique(train_data[:, i][~np.isnan(train_data[:, i])], return_counts=True)
        train_data[:, i][np.isnan(train_data[:, i])] = uniques[np.where(counts == np.amax(counts))][0]
        
    train_data[:, :-1] = contineous_treatment(train_data.T)
    test_data[:, :-1] = contineous_treatment(test_data.T)

    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]

    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    class_labels = np.unique(train_y)

    t = 0

    tree = decision_tree(train_X.T, train_y.T, atribute_labels, t)
    tree.viz("B")
    
    y_pred = tree.predict(test_X, atribute_labels)
    print(confusion_matrix(test_data[:, -1], y_pred, len(class_labels)))
    print(f"{acuraccy_score(test_data[:, -1], y_pred)}% of accuracy")

    all_labels = count_possible_labels(dataset.T)
    k_fold = 3

    print("Holdout Cross Validation: ", holdout_cross_validation(data, atribute_labels, k_fold, t))
    print("Cross Validation: ", k_fold_cross_validation(data, atribute_labels, k_fold, t))
    #print("Leave One Out: ", leave_one_out(data, atribute_labels, t))

def Aplus(dataset, t, k_fold):

    data = dataset.values
    atribute_labels = np.array(list(dataset))
    y_pred,test_data,class_labels= random_forest(dataset,atribute_labels,100)
    print(confusion_matrix(test_data[:, -1], y_pred, len(class_labels)))
    print(f"{acuraccy_score(test_data[:, -1], y_pred)}% of accuracy")
    t=0
    print("Holdout Cross Validation: ", holdout_cross_validation(data, atribute_labels, k_fold, t))
    print("Cross Validation: ", k_fold_cross_validation(data, atribute_labels, k_fold, t))


def main():

    hospitals_dataset = load_data("./data")

    print(f"Shape of the dataset: {hospitals_dataset.shape[0]} rows and {hospitals_dataset.shape[1]} columns")
    print (f"Amount of NaN: \n{hospitals_dataset.isnull().sum()}\n")
    
    k_fold = 3
    
    for i in range(2):
        print(f"############################### t = {i} ###############################")
        print(f"------------------------------- EX: C -------------------------------")
        C(load_data("./data"), i, k_fold)
        print(f"------------------------------- EX: B -------------------------------")
        B(load_data("./data"), i, k_fold)
        print(f"------------------------------- EX: A -------------------------------")
        A(load_data("./data"), i, k_fold)
        print(f"------------------------------- EX: A+ -------------------------------")
        Aplus(load_data("./data"), i, k_fold)
    
    
    # print (f"Hospitals dataset describe:\n {dataset.head()}\n")
    # print(f"How many classes are in y: {np.unique(data[:,-1], return_counts=True)}\n")

    ## Data to prove funcions
    # data_test = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1]])
    # y = np.array([0, 0, 1, 0, 1])
    # x = np.array([[1, 1, 0, 0, 0], #[0, 0, 0, 0, 1], [0, 1, 0, 1, 1]])
    # tree_test = decision_tree(data_test[:,:-1].T, data_test[:,-1].T, ["op. major", "familia", "gran"], 0)
    # print ("desicion tree:")
    # tree_test.viz()
    # y_pred = tree_test.predict(np.array([[1, 0, 1]]), np.array(["op. major", "familia", "gran"]))
    # print ("preduiction a casa: ", y_pred)
    # print("cross_validatiom: ", cross_validation_tree(data_test, np.array(["op. major", "familia", "gran", "casa"]), 2))
    
    # test_data = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1]])
    # test_atributes = np.array(["op. major", "familia", "gran"])
    # tree_test = decision_tree(test_data[:, :-1].T, test_data[:, -1].T, test_atributes, 1)
    # tree_test.viz()
    
    # train_X = contineous_treatment(train_data)
    
    # tree_test = decision_tree(train_X, train_y, attributes, class_labels=class_labels, t=0)
    # tree_test.viz()
    
    # y_pred = tree_test.predict(test_data[:-1, :], attributes, mode=0)
    

    # tree_hospitals = decision_tree(data, target, attributes)
    # print ("prediction heart-disease: ", tree_hospitals.predict(np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1, 1]]), attributes))

    # k_fold = 3
    # print("Holdout Cross Validation", holdout_cross_validation(dataset, attributes, k_fold, all_labels))
    # print("Cross Validation: ", k_fold_cross_validation(dataset, attributes, k_fold, all_labels))
    # print("Leave One Out: ", leave_one_out(dataset, attributes, all_labels))

    # tree_hospitals = decision_tree(data, target, attributes)
    # print ("prediction heart-disease: ", tree_hospitals.predict(np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1, 1]]), attributes))


if __name__ == "__main__":
    main()