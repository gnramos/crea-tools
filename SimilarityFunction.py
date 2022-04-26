# Creating a similarity query function which receives as parameters a text and the number of most similar subjects wanted

from gensim.similarities.docsim import MatrixSimilarity
import gensim
from preprocessingFunctionModule import preprocess
import pandas as pd
import pickle

def search_similarity_query(filename:str, search_document:str, num_best:int=8):
    """ Based on the the given text, it will perform a similarity search with the model already trained and return a Dataframe containing the best results.

    Based on the filename given that define the subjects, this function preprocesses the given text and computes the cossine similarity value for each class (subject) that is stored in the model. These values are then sorted in decreacresing order and stored in a Dataframe, which will finally be returned by the function. 

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).
        search_document: a string type object containing the text that will be used in the similarity query.
        num_best: an integer type object which will limit the number of similar subjects returned. By default, this limit will be 8.
    
    ### Returns:
        A Dataframe type object which will contain the results of the query. This can also be an empty Dataframe if no cossine similarity value is significant enough.
    """

    subjects_df = pd.read_json(filename + '.json')
    subjects_df = subjects_df.sort_values(by=["codigo"])
    subjects_df = subjects_df.reset_index(drop=True)
    try:
        lsi_corpus = gensim.corpora.MmCorpus('AuxFiles\\lsi_model-' + filename + '_mm')
        with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'rb') as f:
            id2wordDict, tfidf_model, lsi_model = pickle.load(f)
    except FileNotFoundError:
        import ModelTraining
        ModelTraining.modelTraining(filename=filename)
        
        lsi_corpus = gensim.corpora.MmCorpus('AuxFiles\\lsi_model-' + filename + '_mm')
        with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'rb') as f:
            id2wordDict, tfidf_model, lsi_model = pickle.load(f)

    cosineSimilarity = MatrixSimilarity(lsi_corpus, num_features = lsi_corpus.num_terms, num_best=num_best)

    # * preprocessing and processing until becomes a matrix of type term_to_topic (V)
    doc = preprocess(search_document)
    query_bow = id2wordDict.doc2bow(doc)
    query_tfidf = tfidf_model[query_bow]
    query_lsi = lsi_model[query_tfidf]

    # * cossine similarity between the vector of the new document vs all other vectors of documents
    # * returns a list of top num_best tuples (id of compared document, similarity)
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
