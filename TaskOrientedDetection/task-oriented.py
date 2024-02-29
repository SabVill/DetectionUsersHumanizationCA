#
# This file provides a classification on unseen data in form of csv dataset for Task oriented CAs
#
import nltk
from nltk.corpus import stopwords
from geotext import GeoText
from nltk.tokenize import sent_tokenize
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import pandas as pd

#  NLP sourses for implementation
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)

#
#   Method: detection of imperative verbs in sentences
#
def sentence_detection(sentences):
    for sentence in sentences:
        tokens = nltk.word_tokenize(str(sentence))
        if tokens[0].lower() in imperative_lexicon:
            return 1
        elif tokens[0].lower() != "please" and tokens[0] != tokens[-1] and tokens[-1] and tokens[
            1].lower() in imperative_lexicon:
            return 1
    return 0


#
# Method: check the presence of sensitive data as time, dates, fiscal code, email adresses
#
def sensitive_detection(text):
    if (bool(re.search('(\d{1,2})[.:](\d{1,2})?([ ]?(am|pm|AM|PM))?', text)) == True) or (
            bool(re.search("^(\\d{1,}) [a-zA-Z0-9\\s]+(\\,)? [a-zA-Z]+(\\,)? [A-Z]{2} [0-9]{5,6}$", text)) == True) or (
            bool(re.search(r'\w*\d\w*', text)) == True) or (
            bool(re.search("\d{2}[- /.]\d{2}[- /.]\d{,4}", text)) == True) or (
            bool(re.search("(([a-zA-Z0-9_+]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))",
                           text)) == True):
        return 1
    return 0


if __name__ == '__main__':

    # Export of proper names lexicon into dictionary
    names_lexicon_pd = pd.read_csv("../dicts/babynames-clean.csv", encoding='utf-8', delimiter=';')
    print(">>> Export names lexicon to dict...")
    firstnames_lexicon = {}
    for index, row in names_lexicon_pd.iterrows():
        firstnames_lexicon[row["Name"]] = row["Gener"]


    # Export of imperative verbs lexicon into dictionary
    imperative_lexicon_pd = pd.read_csv("../dicts/imperativeVB_lexicon_complete.csv", encoding='utf-8', delimiter=';')
    print(">>> Export imperative lexicon to dict...")
    imperative_lexicon = {}
    for index, row in imperative_lexicon_pd.iterrows():
        imperative_lexicon[row["word"]] = row["freq"]

    #
    # ENTER HERE YOUR DATASET
    #
    # It must:
    #       - Has a csv format
    #       - Contain the following column: "turn" -> with users' turns in text format
    #       - Be called "text.csv", or you will have to change the name below
    #
    # Optional: change the delimiter or the number of rows (nrows)
    df = pd.read_csv("test.csv", encoding='utf-8', delimiter=';')

    # Creation of dataframe that will contain all features
    df_features = pd.DataFrame(
        columns=["turn","words", "sentences", "simple_quest", "whq", "imper_quest", "places",
                 "sensitive_data", "relatives", "first_names", "conditional_vb", "gratitude", "thanksgiving", "excuses",
                 "greetings", "self_references", "you_references", "neutral_references", "neg", "pos", "neu"])

    # words to search for in user's turns to detect specific heuristic features
    wh_q = ["what", "why", "who", "where", "when", "how", "whose"]
    first_pron = ["i", "me", "my", "myself", "mine", "we", "our", "ourselves", "us"]
    second_pron = ["you", "your", "yours", "yourself"]
    neutral_pron = ["it", "itself", "its"]
    relatives_names = ["friend", "family", "husband", "wife", "father", "mother", "grandfather", "grandmother", "mommy",
                       "daddy", "daughter", "son", "brother", "sister", "cousin", "daughters", "sons", "brothers",
                       "sisters", "cousins", "aunt", "uncle", "mom", "dad"]
    conditional_verbs = ["'d", "would", "wouldn", "should", "shouldn", "could", "couldn", "might"]
    greetings = ["hi", "hello", "hey", "ehy", "good", "ehi", "bye", "goodbye"]
    time_of_day = ["morning", "afternoon", "evening", "night"]
    gratitude = ["appreciate", "grateful", "appreciated", "support", "cheers", "thankful", "great", "wonderful"]
    thanksgiving = ["thanks", "thank"]
    excuses = ["please", "sorry", "excuse"]

    print(">>> Check for features...")

    id = ""
    for index, row in df.iterrows():

        # print(row)
        simple_quest = 0
        imperative_q = 0
        wh = 0
        quest_mark = 0
        places = 0
        sensitive_data = 0
        conditional_vb = 0
        relatives = 0
        first_names = 0
        first_ref = 0
        second_ref = 0
        neutral_ref = 0
        positive_w = 0
        negative_w = 0
        greetings_presence = 0
        grat = 0
        thanks = 0
        excuse = 0

        tokens = nltk.word_tokenize(row['turn'])

        # Number of words
        words = len([i for i in tokens if i not in punctuations])

        if "?" in row['turn']:
            quest_mark = 1

        # simple question
        if words <= 6 and quest_mark:
            simple_quest = 1

        #  imperative question
        sentences = sent_tokenize(row['turn'])
        imperative_q = sentence_detection(sentences)

        # Presence of places
        if GeoText(row['turn']).cities or GeoText(row['turn']).countries or GeoText(
                row['turn']).country_mentions or GeoText(row['turn']).nationalities:
            places = 1

        # sensitiive data
        sensitive_data = sensitive_detection(row['turn'])

        for token in tokens:

            # Presence of indicative or wh question
            if token.lower() in wh_q and quest_mark:
                wh = 1

            # Presence of relative names
            if token.lower() in relatives_names:
                relatives = 1

            # # Presence of self references
            # if token.lower() in first_pron:
            #     first_second_n = 1

            # Presence of conditional verbs
            if token.lower() in conditional_verbs:
                conditional_vb = 1

            # Presence of gratitude expressions
            if token.lower() in gratitude:
                grat = 1

            # Presence of thanksgiving expressions
            if token.lower() in thanksgiving:
                thanks = 1

            # Presence of excuses expressions
            if token.lower() in excuses:
                excuse = 1

            # Presence of greetings
            if token.lower() in greetings and token.lower() != "good":
                greetings_presence = 1

            if token.lower() == "good" and tokens.index(token) != len(tokens) - 1:
                next_token = tokens[tokens.index(token) + 1]
                if next_token.lower() in time_of_day:
                    greetings_presence = 1

            # Presence of first persons names
            if (token.lower() in firstnames_lexicon and token != "May" and token != "Will"):
                first_names = 1

            # Presence of self references
            if token.lower() in first_pron:
                first_ref = 1

            # Presence of you references
            if token.lower() in second_pron:
                second_ref = 1

            # Presence of it references
            if token.lower() in neutral_pron:
                neutral_ref = 1

        tokens = [w for w in tokens if not w.lower() in stop_words]

        df_features.loc[-1] = [row["turn"],int(words), int(len(sentences)), int(simple_quest), int(wh),
                               int(imperative_q),
                               int(places), int(sensitive_data), int(relatives), int(first_names), int(conditional_vb), int(grat), int(thanks), int(excuse),
                               int(greetings_presence),
                               int(first_ref), int(second_ref), int(neutral_ref),
                               float(sia.polarity_scores(row["turn"])["neg"]),
                               float(sia.polarity_scores(row["turn"])["pos"]),
                               float(sia.polarity_scores(row["turn"])["neu"])]
        df_features.index = df_features.index + 1

    #  preparing features for model classification...
    X_test = df_features[["words", "sentences", "simple_quest", "whq", "imper_quest", "places", "sensitive_data", "relatives", "first_names", "conditional_vb", "gratitude", "thanksgiving", "excuses", "greetings", "self_references", "you_references", "neutral_references", "neg", "pos", "neu"]]
    # import of model
    model_from_joblib = joblib.load('task-oriented.pkl')
    # classification
    y_pred = model_from_joblib.predict(X_test)

    df_def = df_features.assign(output=y_pred)


    #
    # HERE YOU CAN CHANGE THE NAME OF THE RESULTING FILE
    #
    df_def.to_csv("MultiWOZ_testing.csv", index=False)