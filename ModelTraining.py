# Using processed data, train model. Then save similarities in a matrix market doc: 'lsi_model_mm'

import pickle
import gensim
from gensim import corpora

with open('AuxFiles\\documents_df_pickle.txt', 'rb') as f:
    documents_df = pickle.load(f)

lemmatizedData = documents_df["documento"].tolist()

# * gensim dictionary object, which will track each word to its respective id
id2wordDict = corpora.Dictionary(lemmatizedData)

# * gensim doc2bow method to map each word to a integer id and its respective frequency
corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

# * corpus -> list of list of tuples (id of a word, frequency)

tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2wordDict)

lsi_model = gensim.models.LsiModel(tfidf_model[corpus], id2word=id2wordDict, num_topics=100, power_iters=100)

gensim.corpora.MmCorpus.serialize('AuxFiles\\tfidf_model_mm', tfidf_model[corpus])

gensim.corpora.MmCorpus.serialize('AuxFiles\\lsi_model_mm', lsi_model[tfidf_model[corpus]])