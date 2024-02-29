# Automatic Detection of CA's humanization from users

Nowadays, Conversational Agents (CAs) show more and more intelligent and human-like behaviors. A user interacting with them may antropomoprhing them, by ascribing them capabilities that CAs do not actually have. 
This phenomenon can damage the interaction, as the user could be disappointed by CA's mistakes in understanding his/her requests and consequently abandon the conversation.

With this work, we aim to give a practical intrument to automatically detect when the user is excessively humanizing the CA, in order to store this information in a user model that would help the CA to adapt its answers to guide the user to the most correct conversational styles to follow.
The detection is performed on user's (conversational) turns through baseline machine learning models (SVM and Logistic Regression) and from three types of features extracted from texts:

- **Heuristics features**: specific features that determine hat the user is excessively humanizing the CA (from works of the state-of-the-art)
- **Frequency features**: TF-IDF weigths
- **Semantics features**: words embeddings with SentenceBERT

We discriminate the detection for three types of CAs, that are:

- **Task-oriented chatbots**: TaskOrientedDetection
- **QA chatbots**: QADetection
- **LLM-based chatbots**: LLMbasedDetection

For each type of CA, we provide:
- The dataset we used for model(s) training (_datasetname__feat.csv)
- The produced dataset(s) from the automatic detection (_datasetname__(_model_)_testing.csv)
- The program for models training (_CAtype__training_export.py)
- The program for automatic detection over csv dataset of users' turns (_CAtype_.py)
- The program for automatic detection over a single user's turn (_CAtype__singleturn.py)
- The trained model(s) (.pkl files)

## TaskOrientedDetection

Automatic detection of task-oriented CA's humanization ascribed by users.
This model has been trained on the MultiWOZ dataset.


## QADetection

Automatic detection of task-oriented CA's humanization ascribed by users.
This model has been trained on the BBAI dataset.

## LLMbasedDetection

Automatic detection of task-oriented CA's humanization ascribed by users.
This model has been trained on the LMSYS-Chat-1M dataset.

## dicts

This directory contains lexicons expoited for the detection of some features. 

- babynames-clean.csv: first person names
- imperativeVB_lexicon_complete.csv: lexicon of verbs in the infinte/imperative form


