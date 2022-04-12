# Creating a similarity query function which receives as parameters a text and the number of most similar subjects wanted

from gensim.similarities.docsim import MatrixSimilarity
import gensim
from preprocessingFunctionModule import preprocess
from ModelTraining import id2wordDict, tfidf_model, lsi_model
import pandas as pd

# * reading the subjects to compare in the queries
subjects_df = pd.read_json("AuxFiles\\MechatronicsEngeneeringSubjects.json")
subjects_df = subjects_df.sort_values(by=["codigo"])
subjects_df = subjects_df.reset_index(drop=True)

# * loading the already trained lsi model
tfidf_corpus = gensim.corpora.MmCorpus('AuxFiles\\tfidf_model_mm')
lsi_corpus = gensim.corpora.MmCorpus('AuxFiles\\lsi_model_mm')

def search_similarity_query(search_document, num_best=8):

    cosineSimilarity = MatrixSimilarity(lsi_corpus, num_features = lsi_corpus.num_terms, num_best=num_best)

    # * preprocessing and processing until becomes a matrix of type term_to_topic (V)
    doc = preprocess(search_document)
    query_bow = id2wordDict.doc2bow(doc)
    query_tfidf = tfidf_model[query_bow]
    query_lsi = lsi_model[query_tfidf]

    # * cossine similarity between the vector of the new document vs all other vectors of documents
    # * returns a list of top 8 tuples (id of compared document, similarity)
    ranking = cosineSimilarity[query_lsi]

    ranking.sort(key=lambda unit: unit[1], reverse= True)
    result = []

    for subject in ranking:

        result.append (
            {
                'Relevancia': round((subject[1] * 100),6),
                'Código da Matéria': subjects_df['codigo'][subject[0]],
                'Nome da matéria': subjects_df['nome'][subject[0]]
            }

        )
    
    output = pd.DataFrame(result, columns=['Relevancia','Código da Matéria','Nome da matéria'])
    
    return output
