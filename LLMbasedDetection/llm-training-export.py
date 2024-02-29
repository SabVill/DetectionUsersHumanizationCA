
#
# This file present the instructions we used to train our model for the LLM-based CAs dataset turns
#
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from sentence_transformers import SentenceTransformer

# SentenceBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

#
# Method: performs Logistic Regression training
#
def perform_kf_logreg(X, y):
    print(">>> Preparing Logistic Regression Model...")
    logReg = LogisticRegression(solver='liblinear', random_state=42)

    X_train, y_train = X, y

    model = logReg.fit(X_train, y_train)

    # stored the model trained for later uses
    joblib.dump(model, 'llm_logReg.pkl')


#
# Method: performs SVM training
#
def perform_kf_SVM(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    X_train, y_train = X, y

    model = SVM.fit(X_train, y_train)

    # stored the model trained for later uses
    joblib.dump(model, 'llm_svm.pkl')

#
# Method: preprocess of data for training
#
def doit_with_sbert(df):
    tmp = df['turn'].apply(lambda x: sbert.encode(x).flatten()).values
    data = [x for x in tmp]
    tmp_df = pd.DataFrame(data=data, columns=[_ for _ in range(len(data[0]))])

    tmp_df = tmp_df.assign(words=df["words"])
    tmp_df = tmp_df.assign(sentences=df["sentences"])  # attenzione
    tmp_df = tmp_df.assign(simple_quest=df["simple_quest"])
    tmp_df = tmp_df.assign(whq=df["whq"])
    tmp_df = tmp_df.assign(imper_quest=df["imper_quest"])
    tmp_df = tmp_df.assign(places_1=df["places"])
    tmp_df = tmp_df.assign(sensitive_data=df["sensitive_data"])
    tmp_df = tmp_df.assign(relatives=df["relatives"])
    tmp_df = tmp_df.assign(first_names=df["first_names"])
    tmp_df = tmp_df.assign(conditional_vb=df["conditional_vb"])
    tmp_df = tmp_df.assign(gratitude=df["gratitude"])
    tmp_df = tmp_df.assign(thanksgiving=df["thanksgiving"])
    tmp_df = tmp_df.assign(excuses=df["excuses"])
    tmp_df = tmp_df.assign(greetings=df["greetings"])
    tmp_df = tmp_df.assign(self_references=df["self_references"])
    tmp_df = tmp_df.assign(you_references=df["you_references"])
    tmp_df = tmp_df.assign(neutral_references=df["neutral_references"])
    tmp_df = tmp_df.assign(neg=df["neg"])
    tmp_df = tmp_df.assign(pos=df["pos"])
    tmp_df = tmp_df.assign(neu=df["neu"])

    y = df.loc[:, 'Model'].values

    perform_kf_logreg(tmp_df, y)
    perform_kf_SVM(tmp_df, y)



if __name__ == "__main__":

    # takes dataset for training and transforms it into a dataframe
    # LLM_feat is the dataset from BBAI we used for training the model for QA CAs
    df = pd.read_csv('LLM_feat.csv', sep = ',')

    # this is not necessary if the dataset does not have a dialogie_id column
    df = df.drop(columns=['dialogue_id'], axis=1)

    #
    # TRAINING SVM ON SBERT AND HEURISTICS FEATURES
    #
    doit_with_sbert(df)