from nltk.corpus import stopwords,words,brown
from nltk import word_tokenize, sent_tokenize, pos_tag
from string import punctuation
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
import math
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# *OPENING REQUIRED INPUT FILES*
# 
# This is the input data, consisting of the original csv file with the data and the relevant prompts for each question.

all_essays = open('/data/training_set_rel3.csv',encoding='iso-8859-1')
prompt_1 = open('/data/prompt_1.txt',encoding = 'iso-8859-1').read()
prompt_2 = open('/data/prompt_2.txt',encoding = 'iso-8859-1').read()
prompt_3 = open('/data/prompt_3.txt',encoding = 'iso-8859-1').read()
prompt_4 = open('/data/prompt_4.txt',encoding = 'iso-8859-1').read()
prompt_5 = open('/data/prompt_5.txt',encoding = 'iso-8859-1').read()
prompt_6 = open('/data/prompt_6.txt',encoding = 'iso-8859-1').read()
prompt_7 = open('/data/prompt_7.txt',encoding = 'iso-8859-1').read()
prompt_8 = open('/data/prompt_8.txt',encoding = 'iso-8859-1').read()


# *DATA PREPROCESSING*
# 
# This is essentially the extraction of what is to go into the algorithm as input: *set_inp* - a list consisting of (1) scores, (2) essays without English stop words, (3) original essays and (4) essay set number, *prompts* - a list of of the prompt passages in the questions for the various essay sets (with the possiblity to include prompts for all questions at every index), and *word_set* - corpus of English words to check spelling errors against.

stops = set(stopwords.words("english"))
read_all_essays = csv.reader(all_essays)
set_inp = []
for row in read_all_essays:
    if row[1] == '1': #essay set number being trained and tested
        essay_withoutstopwords = [word.lower() for word in word_tokenize(row[2]) if word.lower() not in stops and word not in punctuation]
        essay = row[2].lower()
        score = row[6]
        setno = int(row[1])
        set_inp.append([score,essay_withoutstopwords,essay,setno])
word_list = brown.words()
word_set = set(word_list)
prompts = [0,prompt_1,prompt_2,prompt_3,prompt_4,prompt_5,prompt_6,prompt_7,prompt_8]


# *FUNCTIONS FOR FEATURE AND CLASS EXTRACTION*
# 
# The *featurizer* function is where all the relevant features of the text are extracted, and the *classizer* function extracts the gold standard classes.
# 
# The features being extracted for each essay include:
# 
# - Comparison score between the full essay and the prompt for the corresponding essay set based on tf-idf scores
# - uni-, bi- and trigrams in the full essay and their corresponding tf-idf scores
# - Top 40 words used in the essay
# - POS tags used in the full essay and their counts
# - Lexical density, i.e. count of important POS tags like those for nouns, adjectives and adverbs/all tags
# - Word count, character count, sentence count, paragraph count, number of spelling errors converted into z-scores
# - Type-token ratio, i.e. number of unique words in text/total number of words


def featurizer(set_input,prompt_input, word_set):
    
    features = []
    
    #GENERATION OF TF-IDF SCORES OF UNI-,BI- AND TRIGRAMS
    tfidf_vect = TfidfVectorizer(min_df=1)
    corpus = [item[2] for item in set_input]
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(corpus)
    feature_names = tf.get_feature_names() 
    dense = tfidf_matrix.todense()
    
    #FUNCTION TO GENERATE Z-SCORES
    scaler = StandardScaler()
    
    #COUNTER TO AVOID LENGTH ERROR IN TF-IDF MATRIX
    element = 0
    
    
    #LOOPING THROUGH ESSAYS
    for essay in set_input:
        feature_dictionary = {}
        
        #SIMILARITY SCORE BETWEEN ESSAY AND PROMPT
        prompt = prompt_input[essay[3]]
        tfidf = tfidf_vect.fit_transform([prompt,essay[2]])
        pairwise = tfidf * tfidf.T
        sim = 'sim#' + str(round(pairwise[0,1],1))
        feature_dictionary['sim'] = round(pairwise[0,1],1)
        
        #TF-IDF SCORES OF UNI-,BI- AND TRIGRAMS
        if len(dense) > element:
            current_essay = dense[element].tolist()[0]
            phrase_scores = [pair for pair in zip(range(0, len(current_essay)), current_essay) if pair[1] > 0]
            for i in phrase_scores:
                tf = 'tf#' + str(i[0])
                feature_dictionary[tf] = i[1]
        element += 1
        
        #TOP 40 WORDS
        sorted_top = sorted(Counter(essay[1]), key=Counter(essay[1]).get, reverse=True)[:40]
        for i in sorted_top:
            top = 'top#' + i
            feature_dictionary[top] = 1
        
        #COUNTS OF POS TAGS
        pos_tagged = pos_tag(word_tokenize(essay[2]))
        pos_counts = {}
        important_lexical_items = 0
        all_items = 0
        for item in pos_tagged:
            if item[1] in pos_counts:
                pos_counts[item[1]] += 1
            else:
                pos_counts[item[1]] = 1
            if item[1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS']:
                important_lexical_items += 1
            all_items += 1
        for key, value in pos_counts.items():
            feature_dictionary['POS#' + str(key)] = value 
        
        #LEXICAL DENSITY
        feature_dictionary['lexical_density#'] = important_lexical_items/all_items
        
        #WORD COUNT
        word_count = len(word_tokenize(essay[2]))
        feature_dictionary['wc'] = word_count
        
        #CHARACTER COUNT
        character_count = len(essay[2])
        feature_dictionary['cc'] = character_count
        
        #WORD LENGTH
        word_length = np.average([len(word) for word in word_tokenize(essay[2])])
        feature_dictionary['wl'] = word_length
        
        #SENTENCE COUNT
        sentence_count = len(sent_tokenize(essay[2]))
        feature_dictionary['sc'] = sentence_count
        
        #SENTENCE LENGTH
        avg_sentence_length = np.average([len(sentence.split()) for sentence in sent_tokenize(essay[2])])
        feature_dictionary['sl'] = avg_sentence_length
        
        #PARAGRAPH COUNT
        paragraph_count = len(essay[2].split('\n'))
        feature_dictionary['pc'] = paragraph_count
        
        #NUMBER OF SPELLING ERRORS
        spelling_errors = 0
        for word in essay[2].split():
            if word not in word_set:
                spelling_errors += 1
        feature_dictionary['se'] = spelling_errors
        features.append(feature_dictionary)
        
        #TYPE-TOKEN RATIO
        words_list = []
        unique_words = 0
        for word in essay[2].split():
            if word not in words_list:
                unique_words += 1
                words_list.append(word)
        feature_dictionary['ttr'] = unique_words/word_count

    
    #STANDARD-SCALING NUMERICAL FEATURES AND REPLACING THEM
    
    #LISTS OF ALL ELEMENTS FOR EVERY FEATURE
    word_count_list = []
    character_count_list = []
    word_length_list = []
    sentence_count_list = []
    avg_sentence_length_list = []
    paragraph_count_list = []
    spelling_errors_list = []
    
    for item in features:
        word_count_list.append(item['wc'])
        character_count_list.append(item['cc'])
        word_length_list.append(item['wl'])
        sentence_count_list.append(item['sc'])
        avg_sentence_length_list.append(item['sl'])
        paragraph_count_list.append(item['pc'])
        spelling_errors_list.append(item['se'])
        
    #SCALING EACH LIST
    scaled_word_counts = scaler.fit_transform(word_count_list)
    scaled_character_counts = scaler.fit_transform(character_count_list)
    scaled_word_lengths = scaler.fit_transform(word_length_list)
    scaled_sentence_counts = scaler.fit_transform(sentence_count_list)
    scaled_sentence_lengths = scaler.fit_transform(avg_sentence_length_list)
    scaled_paragraph_counts = scaler.fit_transform(paragraph_count_list)
    scaled_spelling_errors = scaler.fit_transform(spelling_errors_list)
    
    #REPLACING VALUES IN FEATURE LIST WITH SCALED ONES
    j = 0
    for wc in scaled_word_counts:
        features[j]['wc'] = wc
        j += 1
    j = 0
    for cc in scaled_character_counts:
        features[j]['cc'] = cc
        j += 1
    j = 0
    for wl in scaled_word_lengths:
        features[j]['wl'] = wl
        j += 1
    j = 0
    for sc in scaled_sentence_counts:
        features[j]['sc'] = sc
        j += 1
    j = 0
    for sl in scaled_sentence_lengths:
        features[j]['sl'] = sl
        j += 1
    j = 0
    for pc in scaled_paragraph_counts:
        features[j]['pc'] = pc
        j += 1
    j = 0
    for se in scaled_spelling_errors:
        features[j]['se'] = se
        j += 1
    return features
    


def classizer(input_set): #to get classes   
    classes = [l[0] for l in input_set]
    return classes


# *SPLITTING DATA, STORING VECTORIZED FEATURES AND ALSO CLASSES*

set_inp_train, set_inp_test = train_test_split(set_inp, test_size=0.1, random_state=1)

vectorizer = DictVectorizer(sparse = True)

features_train = vectorizer.fit_transform(featurizer(set_inp_train,prompts,word_set)) #vectorizing feature dictionary

classes_train = classizer(set_inp_train)


# *TRAINING AND TESTING ALGORITHM*
# 
# The type of classifier used is Support Vector Machine (SVM). Two types of accuracies are calculated: one strict accuracy, which measures how many predicted classes matched the actual classes exactly, and one lenient accuracy, which allows for an error of 1 point.


classifier = svm.LinearSVC()

classifier.fit(features_train,classes_train)

features_test = vectorizer.transform(featurizer(set_inp_test,prompts,word_set))

classes_test = classizer(set_inp_test)

predicted_classes = classifier.predict(features_test) #predicting classes for test set

acc = accuracy_score(classes_test, predicted_classes)*100 #accuracy of predicted classes

print('The strict accuracy of this classifier is ' + str(("%.2f" % acc)) + '%.')

total = 0
correct = 0
for output in zip(classes_test,predicted_classes):
    if (int(output[0]) + 1) == int(output[1]) or (int(output[0]) - 1) == int(output[1]) or output[0] == output[1]:
        correct += 1
    total += 1
    
accur = correct*100/total

print('The lenient accuracy of this classifier is ' + str(("%.2f" % accur)) + '%.')