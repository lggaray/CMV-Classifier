import pickle, aux
from collections import defaultdict
from textblob import TextBlob
import spacy

'''
aux.py is auxiliar functions file,
which contains the following functions:
    *disambiguate
    *find_struct
    *take_n_first
    *take_n_last
'''

######################### LOADING PICKLES ###################################
with open('../dataset/neg_final', 'rb') as f:
    pair_of_comments = pickle.load(f)

with open('../selfmade_resources/neg_rules', 'rb') as f:
    neg_rules = pickle.load(f)

with open('../selfmade_resources/pos_rules', 'rb') as f:
    pos_rules = pickle.load(f)

with open('../selfmade_resources/contrast_dm', 'rb') as f:
    contrast_dm = pickle.load(f)

with open('../selfmade_resources/modals_dict', 'rb') as f:
    modals_dict = pickle.load(f)

with open('../selfmade_resources/discourse_markers', 'rb') as f:
    discourse_markers = pickle.load(f)

nlp = spacy.load('en')

############################ CREATING THE DICT ################################

'''
building the dictionary with the features we had choosen
we use some dictionaries and lists we made for this task
textblob and spacy helped with a couple of features
dicc is a dict of dict, where the inner dict i contains
the features of the 2-tuple i.
'''

dicc = defaultdict(dict)
for i,(father,son) in enumerate(pair_of_comments):
    new_dict = defaultdict(int)
    dicc[i] = new_dict
    #-----------DISCOURSE MARKERS & MODALS--------------------#
    for key in modals_dict.keys():
        if key in father:
            new_dict['Fmd|' + modals_dict[key]] += 1
        if key in son:
            new_dict['Smd|' + modals_dict[key]] += 1
    for dm in discourse_markers:
        if dm in son:
            new_dict['Sdm|' + dm] += 1
    #--------------MODALS DISAMBIGUATION----------------------#
    txt = [TextBlob(father), TextBlob(son)] 
    '''
    txt[0] is the tokenized father
    txt[1] is the tokenized son
    '''
    desam0 = aux.disambiguate(txt[0].words)
    if desam0 != []:
        for d in desam0:
            new_dict['Fmd|'+d] += 1 
    desam1 = aux.disambiguate(txt[1].words)
    if desam1 != []:
        for d in desam1:
            new_dict['Smd|'+d] += 1
    #-----------------SON LENGTH------------------------------#
    n_words_son = len(txt[1].words)
    new_dict['SnWords'] = n_words_son  
    #--------------------RULES--------------------------------#
    for neg_rule in neg_rules:
        if neg_rule in txt[1]:
            new_dict['SNegRule'] = 1
            break
    for pos_rule in pos_rules:
        if pos_rule in txt[1]:
            new_dict['SPosRule'] = 1
            break 
    #------------------VERBS & N_FIRST------------------------#
    for i,text in enumerate(txt):
        '''
        we use the lemmatized text later for taking
        the n_first/last words
        '''
        lemmatized_text = []
        for word in text.words: 
            lemma = nlp(word.string)[0].lemma_
            pos = nlp(word.string)[0].pos_ 
            if pos == 'VERB':
                if i == 0: #father
                    new_dict['FVerb|'+lemma] += 1
                else: #son
                    new_dict['SVerb|'+lemma] += 1
            lemmatized_text.append(lemma)
        if i == 1:
            for j in range(3):
                new_dict['Sn_first|' + aux.take_n_first(j+1, " ".join(lemmatized_text))] += 1
                new_dict['Sn_last|' + aux.take_n_last(j+1, " ".join(lemmatized_text))] += 1
        else:
            for j in range(3):
                new_dict['Fn_first|' + aux.take_n_first(j+1, " ".join(lemmatized_text))] += 1
                new_dict['Fn_last|' + aux.take_n_last(j+1, " ".join(lemmatized_text))] += 1
    #-----------------SIMILARITY----------------------------#
    doc1 = nlp(father)
    doc2 = nlp(son)
    new_dict['Similarity'] = doc1.similarity(doc2)
    #--------------------NGRAMS-----------------------------#
    for j in range(3):
        res = []
        res = aux.compare(txt[0].ngrams(j), txt[1].ngrams(j))
        if res != []:
            new_dict[str(j)+'gram'] = len(res)
   
'''
we add this just to keep in track
the length of the dataset
''' 
dicc['dat'] = {'len_conjunto': len(pair_of_comments)}

'''
save the dict
'''
with open('set_neg15', 'wb') as f:
    pickle.dump(dicc, f)
