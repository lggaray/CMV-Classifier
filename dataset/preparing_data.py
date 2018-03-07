# coding: utf-8

import praw, re, pickle
from collections import defaultdict
from pathlib import Path

'''
Setting the API configuration
'''
reddit = praw.Reddit(client_id='aWlbo3TvlEzzlg',
                     client_secret='0NFOZYPIddB6qRIZhP-d6Qcx140',
                     username='luxion95',
                     password='lala123',
                     user_agent='luxion95v1')


subreddit = reddit.subreddit('changemyview')
hot_cmv = subreddit.top(limit=10000)

'''
Here we have two lists:
    * comments_id is a list with all the coments
    * id_to_comemnt is a dict, given an ID returns
      the comment associated with it
'''
comments_id = []
id_to_comment = {}
number_of_submissions = 0
for submission in hot_cmv:
    if not submission.stickied:
        number_of_submissions += 1
        submission.comments.replace_more(limit=0)
        comments_id.append(submission.comments.list())
        text = submission.selftext
        sanitized_post = text.partition('*This is a footnote')
        id_to_comment[str(submission)] = sanitized_post[0]


'''
Filter the hidden comments
'''
for id_list in comments_id:
    for idd in id_list:
        sanitized_body = idd.body.partition('This is hidden text for')
        id_to_comment[str(idd)] = str(sanitized_body[0])

'''
Here we replace quotes in comments 
for the reserved word "quotex"
'''
for key,value in id_to_comment.items():
    text = value.split('\n')
    for i,t in enumerate(text):
        if t != '' and t[0] == '>':
            text[i] = ' quotex '
    id_to_comment[key] = "".join(text)

'''
Sanitizing the comments
'''
def cleaning(text):
    a = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'linkx', text)
    b = re.sub("'", "", a)
    c = re.sub("’", "", b)
    d = re.sub("´", "", c)
    e = re.sub(r"[^a-zA-Z0-9.]+", ' ', d)
    f = re.sub('_', '', e)
    g = re.sub(' +',' ',f)
    h = g.rstrip().lstrip() 
    return h

for key,value in id_to_comment.items():
    id_to_comment[key] = cleaning(value)


'''
For each reddit post, we build a comment tree
'''
dicc = defaultdict(dict)
for id_list in comments_id:
    for comment in id_list:
        aux = {}
        if not str(comment.parent()) in dicc:
         dicc[str(comment.parent())] = aux
         aux[comment.id] = comment.body
        else:
         a = dicc[str(comment.parent())]
         if comment.id in a:
             a[comment.id] += comment.body
         else:
             a[comment.id] = comment.body

'''
Define a DFS look up for the trees
'''
def dfs_paths(graph, start):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph[vertex].keys()) - set(path):
            if next not in graph:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

'''
Here we create a list of the top level IDs
and then build a nested comments dicc, which
follows the tree structure
'''
list_id_top_level = []
for i in range(number_of_submissions):
    list_id_top_level.append(list(id_to_comment.keys())[i])

nested_comments_dicc = defaultdict(list)
for id_top_level in list_id_top_level:
    nested_comments_dicc[id_top_level] = list(dfs_paths(dicc,id_top_level))


'''
Final is a list of lists where each inner list is
a complete DFS path, from the top level comment
to the final comment of that thread, but first
we check that comment is not removed or deleted
'''
final = []
auxi = []
numb_of_threads = 0
for key,value in nested_comments_dicc.items():
    for comment_list in value:
        for comment in comment_list:
            if 'removed' not in id_to_comment[comment] and 'deleted' not in id_to_comment[comment]:
                auxi.append(id_to_comment[comment].lower())
        if len(auxi) > 1:
            final.append(auxi)
            auxi = []
            numb_of_threads += 1
        elif len(auxi) == 1:
            auxi = []

'''
building a set of 2-tuples, following the tree structure
of each reddit post
'''
couple_of_comments = set()
for nested_comments in final:
    for v,w in zip(nested_comments[:-1], nested_comments[1:]):
        couple_of_comments.add((v,w))

'''
delteadores: set of 2-tuple, which the first one
             is the comment who got the delta, 
             and the second one is the comment
             who gave the delta.
             This set represents support relations
'''
delteadores = set()
for a in final:
    for i,b in enumerate(a):
        if "confirmed 1 delta awarded to" in b:
            delteadores.add((a[i-2],a[i-1]))
            del a[i]
        elif "this delta has been rejected" in b:
            del a[i]

################## POSITIVES ###########################
'''
2-tuples set with each post and first-level-comment 
'''
pos_rules_ex = set()
for thread in final:
    if len(thread) > 2:
        pos_rules_ex.add((thread[0],thread[1]))

################# NEGATIVES ###########################
'''
make sure any bot-comment is in our set
'''
fails = set()
for fst,snd in delteadores:
    if "confirmed 1 delta awarded to" in fst or "this delta has been rejected" in fst:
           fails.add((fst,snd))

delta_neg = auxx - fallutos

'''
extract pair of comments where there is a neg rule
'''
with open('neg_rules', 'rb') as f:
    neg_rules = pickle.load(f)

neg_rules_ex = set()
for fst,snd in couple_of_comments:
    for rule in neg_rules:
            if rule in snd:
                neg_rules_ex.add((fst, snd))

'''
count number of examples taken with deltas and rules
in each case (positive and negative). Also count the
number of submissions 
'''
nCountDelta = "neg deltas" + " " + str(len(delta_neg))
pCountRules = "pos rules" + " " + str(len(pos_rules_ex))
nCountRules = "neg rules" + " " +str(len(neg_rules_ex))
noSubmissions = "number of submissions" + " " +str(number_of_submissions)

'''
pickling time!
'''
data_pickle = Path('data')
if data_pickle.is_file():
    with open('data', 'rb') as f:
        datos = pickle.load(f)
else:
    datos = defaultdict(list)

datos["top"] = [nCountDelta, pCountRules, noSubmissions]

with open('data', 'wb') as f:
    pickle.dump(datos, f)

with open('pos_rules_ex_top', 'wb') as f:
    pickle.dump(pos_rules_ex, f)

with open('neg_delta_ex_top', 'wb') as f:
    pickle.dump(delta_neg, f)

with open('neg_rules_ex_controversial', 'wb') as f:
    pickle.dump(neg_rules_ex, f)
