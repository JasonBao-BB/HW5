from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

# open file and read it
str1 = open("text1", "r").read().lower()
str2 = open("text2", "r").read().lower()

# construct a 2-d array
corpus = [str1, str2]

vector = CountVectorizer(analyzer="word", ngram_range=(1, 3))

X = vector.fit_transform(corpus)

X_vector = X.toarray()

# print X.toarray()
print("result to array is :")
print(X_vector)

# get features name
print("get features name :")
print(vector.get_feature_names())
print()

# calculate cosine similarity
print("cosine similarity by sklearn is :")
print(cosine_similarity(X_vector))

# calculate jaccard score
print("jaccard score by sklearn is :")
print(jaccard_score(X_vector[0], X_vector[1], average=None))


