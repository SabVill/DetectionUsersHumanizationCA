# Automatic Detection of User's Humanization of Chatbots

With this work, we aim to give a practical instrument to automatically detect when the user is excessively humanizing a chatbot.
The detection is performed on the user's (conversational) turns through baseline machine learning models (SVM and Logistic Regression) and from three types of features extracted from texts:

- **Heuristics features**: specific features that determine that the user is excessively humanizing the chatbot (from works of the state-of-the-art)
- **Frequency features**: TF-IDF weights
- **Semantics features**: SentenceBERT words embeddings 

We discriminate the detection for three types of chatbots, that are:

- **Task-oriented chatbots**: TaskOrientedDetection
- **Q&A chatbots**: QADetection
- **LLM-based chatbots**: LLMbasedDetection

For each type of chatbot, we provide:
- The dataset we used for model(s) training (_datasetname_ _feat.csv)
- The produced dataset(s) from the automatic detection (_datasetname_ _(_model_)_testing.csv)
- The program for models training (_CAtype_ _training_export.py)
- The program for automatic detection over csv dataset of users' turns (_CAtype_.py)
- The program for automatic detection over a single user's turn (_CAtype_ _singleturn.py)
- The trained model(s) (.pkl files)

Libraries and tools used for the detection:
- **sklearn** for vectorization, models training and test
- **pandas** for dataframes
- **nltk** for NLP techniques
- **joblib** for models export and import
- **SentenceTransformer** for S-BERT word embeddings

## TaskOrientedDetection

Automatic detection of task-oriented chatbot's humanization ascribed by users.
This model has been trained on the MultiWOZ dataset.


## QADetection

Automatic detection of task-oriented chatbot's humanization ascribed by users.
This model has been trained on the BBAI dataset.

## LLMbasedDetection

Automatic detection of task-oriented chatbot's humanization ascribed by users.
This model has been trained on the LMSYS-Chat-1M dataset.

## dicts

This directory contains lexicons exploited for the detection of some features. 

- babynames-clean.csv: first person names
- imperativeVB_lexicon_complete.csv: lexicon of verbs in the infinite/imperative form


