# Using processed data, train model. Then save similarities in a matrix market doc: 'lsi_model_mm'

import pickle
import gensim
from gensim import corpora

def modelTraining(filename:str):
    """ Based on the the given file for the subjects, reads the processed data and trains a tf-idf model and also a lsi-model.

    This function reads the .json file that contains all subjects, which will become the classes for the classification function. Moreover, it will read the saved processed data and train the models. The training consists in, firstly, creating a dictionary that will map all words to an numerical id. Secondly, it will create a bag-of-words corpus that will contain all the words frequency. After that, this values will transformed with the tf-idf model, which will atribute more value to important words that appears less in documents but is very meaningful to its topic. Finally, the values processed by tf-idf will be again transformed using LSI, which performs a SVD (singular value decomposition) and creates a document-to-topic matrix that will saved in a Matrix Market format and be used in the similarity query function. This function also creates a pickle file that will store id2wordDict, tfidf_model, lsi_model objects.

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).
    
    ### Returns:
        None
    """

    try:
        with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'rb') as f:
             documents_df = pickle.load(f)
    except FileNotFoundError:
        import DataPreprocessing
        DataPreprocessing.dataPreprocessing(filename=filename)

        with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'rb') as f:
            documents_df = pickle.load(f)

    lemmatizedData = documents_df["documento"].tolist()

    # * gensim dictionary object, which will track each word to its respective id
    id2wordDict = corpora.Dictionary(lemmatizedData)

    # * gensim doc2bow method to map each word to a integer id and its respective frequency
    corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

    # * corpus -> list of list of tuples (id of a word, frequency)

    tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2wordDict)

    # * num_topics = numbers of subjects
    lsi_model = gensim.models.LsiModel(tfidf_model[corpus], id2word=id2wordDict, num_topics=len(lemmatizedData), power_iters=100)

    gensim.corpora.MmCorpus.serialize('AuxFiles\\tfidf_model-' + filename + '_mm', tfidf_model[corpus])

    gensim.corpora.MmCorpus.serialize('AuxFiles\\lsi_model-' + filename + '_mm', lsi_model[tfidf_model[corpus]])

    with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'wb') as f:
        pickle.dump( [id2wordDict, tfidf_model, lsi_model], f)