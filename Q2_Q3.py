import nltk
from nltk.corpus import brown
from pickle import dump
from pickle import load

# get words with tags
brown_tagged_sents = brown.tagged_sents(categories='news')

# 90% data as training part
size = int(len(brown_tagged_sents) * 0.9)
# training data 90%
train_sents = brown_tagged_sents[:size]
# test data 10%
test_sents = brown_tagged_sents[size:]

# use UnigramTagger
t1 = nltk.UnigramTagger(train_sents)
res1 = t1.evaluate(test_sents)

# use BigramTagger
t2 = nltk.BigramTagger(train_sents)
res2 = t2.evaluate(test_sents)
# use TrigramTagger
t3 = nltk.TrigramTagger(train_sents)
res3 = t3.evaluate(test_sents)
# Compare the performances with 3 taggers
print(res1, " ", res2, " ", res3)


############################################################################################################
############################################################################################################
########################################### Storing All Taggers ############################################
############################################################################################################
############################################################################################################

# storing unigram tagger
output_t1 = open('t1.pkl', 'wb')
dump(t1, output_t1, -1)
output_t1.close()
print("T1 saved success")

# storing trigram tagger
output_t2 = open('t2.pkl', 'wb')
dump(t2, output_t2, -1)
output_t2.close()
print("T2 saved success")

# storing trigran tagger
output_t3 = open('t3.pkl', 'wb')
dump(t3, output_t3, -1)
output_t3.close()
print("T3 saved success")


############################################################################################################
############################################################################################################
####################################### Tagger words from text1 ############################################
############################################################################################################
############################################################################################################

# open all taggers from local
input_t1 = open('t1.pkl', 'rb')
input_t2 = open('t2.pkl', 'rb')
input_t3 = open('t3.pkl', 'rb')

# load all taggers
tagger_t1 = load(input_t1)
tagger_t2 = load(input_t2)
tagger_t3 = load(input_t3)

# tagged words from text1

text = open("text1").read()
tokens = text.split()
# add tags to text1
tag_res_t1 = tagger_t1.tag(tokens)
tag_res_t2 = tagger_t2.tag(tokens)
tag_res_t3 = tagger_t3.tag(tokens)
# show the results
print(tag_res_t1)
print(tag_res_t2)
print(tag_res_t3)

print()
# Can lowercase affect the performance of the POS tagger?
text_lower = open("text1").read().lower()
tokens_lower = text.split()

tag_res_t1_lower = tagger_t1.tag(tokens_lower)
tag_res_t2_lower = tagger_t2.tag(tokens_lower)
tag_res_t3_lower = tagger_t3.tag(tokens_lower)

print(tag_res_t1_lower)
print(tag_res_t1_lower)
print(tag_res_t1_lower)

# It depends on the word, if the word is an old world appeared before, it has less influence no matter it is lower case
# or upper case. However, if we want to tag a new word, the lower case could has a better performance than upper case,
# the reason is uppercase could have different meaning based on context.

# Problem 3. What are some of the possible NLP components of a Question Answering System?
# a. question classification
# b. information retrieval
# c. answer extraction
# These components play a essential role in QAS. Question classification play
# primary role in QA system to categorize the question based upon on the type of its entity