#
# This file present the instructions we used to train our model for the QAs dataset turns
#
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


#
# Method: performs Logistic Regression training
#
def perform_logreg(X, y):
    print(">>> Preparing Logistic Regression Model...")
    logReg = LogisticRegression(solver='liblinear', random_state=42)

    X_train, y_train = X, y

    model = logReg.fit(X_train, y_train)

    # stored the model trained for later uses
    joblib.dump(model, 'qas_logReg.pkl')


#
# Method: performs SVM training
#
def perform_kf_SVM(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    X_train, y_train = X, y

    model = SVM.fit(X_train, y_train)

    # stored the model trained for later uses
    joblib.dump(model, 'qas_svm.pkl')


#
# Method: preprocess of data for training
#
def doit_with_tfidf(df):

    # TFIDF vectorization of "turn" column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["turn"].iloc[::-1])

    # append all the remaining heuristics features
    defin = pd.DataFrame(X.toarray(),
                         columns=vectorizer.get_feature_names())

    defin = defin.assign(words=df["words"])
    defin = defin.assign(sentences=df["sentences"])
    defin = defin.assign(simple_quest=df["simple_quest"])
    defin = defin.assign(whq=df["whq"])
    defin = defin.assign(imper_quest=df["imper_quest"])
    defin = defin.assign(places_1=df["places"])
    defin = defin.assign(sensitive_data=df["sensitive_data"])
    defin = defin.assign(relatives=df["relatives"])
    defin = defin.assign(first_names=df["first_names"])
    defin = defin.assign(conditional_vb=df["conditional_vb"])
    defin = defin.assign(gratitude=df["gratitude"])
    defin = defin.assign(thanksgiving=df["thanksgiving"])
    defin = defin.assign(excuses=df["excuses"])
    defin = defin.assign(greetings=df["greetings"])
    defin = defin.assign(self_references=df["self_references"])
    defin = defin.assign(you_references=df["you_references"])
    defin = defin.assign(neutral_references=df["neutral_references"])
    defin = defin.assign(neg=df["neg"])
    defin = defin.assign(pos=df["pos"])
    defin = defin.assign(neu=df["neu"])

    perform_kf_SVM(defin, df["Model"])

#
# Method: returns a ordered lists of features for the future TF-IDF weigths
#
# Its usage is necessary as features need to be organized in the same order of the training model.
# If the model is trained on new data, it is necessary to run again this method
# otherwise, it is not necessary
#
def tfidf(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["turn"])
    names = vectorizer.get_feature_names()

    features = pd.DataFrame(columns=["feature_name"])

    features["feature_name"] = names

    features.to_csv("qas_tfidf_features.csv", index=False)


if __name__ == "__main__":

    # takes dataset for training and transforms it into a dataframe
    # BBAI_feat is the dataset from BBAI we used for training the model for QA CAs
    df = pd.read_csv('BBAI_feat.csv', sep = ',')

    # this is not necessary if the dataset does not have a dialogie_id column
    df = df.drop(columns=['dialogue_id'], axis=1)

    # use the line belowe only if a new dataset is used for training
    # tfidf(df)


    #
    # TRAINING SVM ON TFIDF AND HEURISTICS FEATURES
    #
    print(">>> Performing TFIDF vectorization...")
    doit_with_tfidf(df)

    #
    # TRAINING LOGISTIC REGRESSION ON HEURISTICS FEATURES
    #
    X = df[["words", "sentences", "simple_quest", "whq", "imper_quest", "places", "sensitive_data", "relatives",
            "first_names", "conditional_vb", "gratitude", "thanksgiving", "excuses", "greetings", "self_references",
            "you_references", "neutral_references", "neg", "pos", "neu"]]
    X = np.array(X)
    y = df["Model"]
    y = np.array(y)

    perform_logreg(X, y)