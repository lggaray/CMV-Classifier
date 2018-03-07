import pickle

with open('../selfmade_resources/neg_rules', 'rb') as f:
    neg_rules = pickle.load(f)
with open('../selfmade_resources/contrast_dm', 'rb') as f:
    contrast_dm = pickle.load(f)


'''
Catch n first words of a text, with some excpetions
Take n and a text STRING
Return a list
'''
def  take_n_first(n, text):
    clean1 = text.replace('quotex', '')
    clean2 = clean1.replace('linkx', '')
    splitted_text = clean2.split()
    res = " ".join(splitted_text[:n])
    return res


'''
Catch n last words of a text, with some excpetions
Take n and a text STRING
Return a list
'''
def  take_n_last(n, text):
    clean1 = text.replace('quotex', '')
    clean2 = clean1.replace('linkx', '')
    clean3 = clean2.replace('.', '')
    clean4 = clean3.replace('8710', '')
    clean5 = clean4.replace('cmv', '')
    clean6 = clean5.replace('delta', '')    
    splitted_text = clean6.split()
    return ' '.join(splitted_text[-n:])


'''
Find a contrast structure in a sentence
Take a sentence
Return a boolean, telling if the struct was find or not
'''
def find_struct(sentence):
    x,y = set(), set()
    indexX, indexY = 0,0
    res = False
    splitted_sentence = sentence.words
    for neg_rule in neg_rules_cleaned:
        if neg_rule in sentence:
            splitted_rule = neg_rule.split()
            indexX = splitted_sentence.index(splitted_rule[-1])
            x.add(neg_rule)
            for contrast in contrast_dm:
                if contrast in splitted_sentence:
                    indexY = splitted_sentence.index(contrast)
                    y.add(contrast)
        if x.intersection(y) == set() and indexX < indexY:
            res = True
            break
    return res


'''
Look for identical n_grams in two texts
Take two n_grams list
Return a list of matching grams
'''
def compare(ngrams1, ngrams2):
    common = []
    done = []
    for grams1 in ngrams1:
        if grams1 in ngrams2 and grams1 not in done:
            common.append(grams1)
            done.append(grams1)
    return common


'''
Solve ambiguity problems with modals, case NOT
Take a comment, as a list of words
Return a list of disambiguate modals
'''
def disambiguate(text):
    res = []
    for i,word in enumerate(text):
        if word == 'may' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('may not')
            else:
                res.append('may')
        if word == 'have' and i+1 <= len(text)-1:
            if text[i+1] == 'to' and text[i-1] == 'not' :
                res.append('not have to')
            elif text[i+1] == 'to':
                res.append('have to')
        if word == 'able' and text[i+1] == 'to':
            if text[i-1] == 'not':
                res.append('cant')
            else:
                res.append('can')
        if word == 'can' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('cant')
            else:
                res.append('can')
        if word == 'could' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('cant')
            else:
                res.append('can')
        if word == 'might' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('mightnt')
            else:
                res.append('might')
        if word == 'must' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('mustnt')
            else:
                res.append('must')
        if word == 'need' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('neednt')
            else:
                res.append('need')
        if word == 'ought' and i+1 <= len(text)-1:
            if text[i+1] == 'not' and text[i+1] == 'to':
                res.append('oughtnt to')
            elif text[i+1] == 'to':
                res.append('ought to')
        if word == 'shall' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('shant')
            else:
                res.append('shall')
        if word == 'should' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('shouldnt')
            else:
                res.append('should')
        if word == 'will' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('wont')
            else:
                res.append('will')
        if word == 'would' and i+1 <= len(text)-1:
            if text[i+1] == 'not':
                res.append('wouldnt')
            else:
                res.append('would')
    return res
