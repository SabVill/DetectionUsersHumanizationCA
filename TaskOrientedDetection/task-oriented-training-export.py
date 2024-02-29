#
# This file present the instructions we used to train our model for the task oriented dataset turns
#
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib

#
# Method: Performs SVM training
#
def SVM_classification(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    X_train, y_train = X, y

    model = SVM.fit(X_train, y_train)

    # stored the model trained for later uses
    joblib.dump(model, 'task-oriented.pkl')



if __name__ == "__main__":

    # takes dataset for training and transforms it into a dataframe
    # MultiWOZ_feat is the dataset from MultiWOZ we used for training the model for task-oriented CAs
    df = pd.read_csv('MultiWOZ_feat.csv')

    # takes the heuristics features as features for the training and the column Model as the goldstandard (humans annotations)
    X = df[["words", "sentences", "simple_quest", "whq", "imper_quest", "places", "sensitive_data", "relatives", "first_names", "conditional_vb", "gratitude", "thanksgiving", "excuses", "greetings", "self_references", "you_references", "neutral_references", "neg", "pos", "neu"]]
    X = np.array(X)
    y = df["Model"]
    y = np.array(y)


    SVM_classification(X, y)




