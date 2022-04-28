from gensim import corpora
from gensim.similarities.docsim import MatrixSimilarity
from nltk.corpus import stopwords
import gensim
import os
import pandas as pd
import pickle
import spacy

# DataPreprocessing.py ########################################################

def dataPreprocessing(filename:str):
    """ Reads the given file, processes the texts and saves it as a Dataframe into a pickle file.

    This function reads the .json file that contains all subjects and, for each row, concatenates the name, syllabus and content into a single string. This text will then be processed by the preprocess function inserted into a Dataframe. Finally, this Dataframe is saved in a pickle binary text format.

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).

    ### Returns:
        None
    """

    subjects_df = pd.read_json(filename + '.json')
    subjects_df = subjects_df.sort_values(by=["codigo"])
    subjects_df = subjects_df.reset_index(drop=True)

    documents_list = []

    for i, row in subjects_df.iterrows():

        # * reading values of each subject (row)
        subject_id = row["codigo"]
        name = row["nome"]
        syllabus = row["ementa"]
        content = row["conteudo"]

        # * combining them to create the subject document
        text = name + ' ' + syllabus + ' ' + content

        # * preprocessing
        preProcessedText = preprocess(text)
        documents_list.append(preProcessedText)

    documents_series = pd.Series(documents_list, name="documento")

    documents_df = pd.concat([subjects_df, documents_series], axis=1)

    with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'wb') as f:
        pickle.dump(documents_df, f)


# ModelTraining.py ########################################################

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
        dataPreprocessing(filename=filename)

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

# preprocessingFunctionModule.py ##############################################
nlp = spacy.load('pt_core_news_lg')

# * adding custom texts that dont represent real words (noises)
noises_list = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi"]

stopWords_list = stopwords.words("portuguese")

# * adding custom words to StopWords list
stopWords_list += [
    'referente',
    'seguinte',
    'etc',
    'ª',
    'tal',
    'um',
    'dois',
    'tres',
    'vs',
    'aula',
    'tal',
]

# * preprocessing stopwords to correct format
stopWords_list = gensim.utils.simple_preprocess(" ".join(stopWords_list), deacc=True, min_len=1, max_len=40)

# * manual intervention, changing final lemmas
intervention_dict = {
    "campar": "campo",
    "seriar":"serie",
    "eletromagnetico":"eletromagnetismo",
}

def preprocess(text:str):
    """ Preprocesses a given text and returns a list of processed words.

    This function firstly uses the simple_preprocess function from Gensim and removes predefined strings that are considered noises. After that, the function uses the pipeline from Spacy, which has a tokenizer, tagger and parser. Then, it removes stopwords and all words are lemmatized by Spacy. Finally, some predefined lemmas are changed by a dictionary and the remaining lemmas are returned as a list.

    ### Parameters:
        text: a string type object containing the text to be processed.

    ### Returns:
        A list type object containing all words as lemmas.
    """

    # * importing stopwords from nltk and spacy pipeline
    global nlp
    global stopWords_list
    global noises_list
    global intervention_dict

    # * preprocessing text with gensim.simple_preprocess, eliminating noises: lowercase, tokenized, no symbols, no numbers, no accents marks(normatize)
    text_list = gensim.utils.simple_preprocess(text, deacc=True, min_len=1, max_len=40)

    # * recombining tokens to a string type object and removing remaining noises
    text_str = " ".join([word for word in text_list if word not in noises_list])

    # * preprocessing with spacy, retokenizing -> tagging parts of speech (PoS) -> parsing (assigning dependencies between words) -> lemmatizing
    text_doc = nlp(text_str)

    # * re-tokenization, removing stopwords and lemmatizing
    lemmatized_text_list = [token.lemma_ for token in text_doc if token.text not in stopWords_list]

    # * manual intervention conversion of lemmas and removing 1 letter stopwords
    output = []
    for token in lemmatized_text_list:
        if len(token) <= 1:
            continue
        if token in intervention_dict:
            output.append(intervention_dict[token])
        else:
            output.append(token)

    return output


def search_similarity_query(course, query, num_best=8):
    '''Apply the course model to search for query within subjects.

    Keyword arguments:
    course -- a string containing the name of course.
    query -- a string containing the text to be searched for.
    num_best -- the number of similar subjects shown.

    Returns:
        A DataFrame with the query results.
    '''

    subjects_df = pd.read_json(f'{course}.json')
    subjects_df = subjects_df.sort_values(by=['codigo'])
    subjects_df = subjects_df.reset_index(drop=True)

    lsi_corpus_file = f'AuxFiles\\lsi_model-{course}_mm'
    if not os.path.isfile(lsi_corpus_file):
        modelTraining(course)
    lsi_corpus = gensim.corpora.MmCorpus(lsi_corpus_file)

    file = f'AuxFiles\\dict_and_models_pickle-{course}.txt'
    with open(file, 'rb') as f:
        id2wordDict, tfidf_model, lsi_model = pickle.load(f)

    cosineSimilarity = MatrixSimilarity(lsi_corpus,
                                        num_features=lsi_corpus.num_terms,
                                        num_best=num_best)

    doc = preprocess(query)
    query_bow = id2wordDict.doc2bow(doc)
    query_tfidf = tfidf_model[query_bow]
    query_lsi = lsi_model[query_tfidf]

    ranking = sorted(cosineSimilarity[query_lsi],
                     key=lambda unit: unit[1], reverse=True)
    result = [{'Relevancia': round((subject[1] * 100), 6),
               'Código da Matéria': subjects_df['codigo'][subject[0]],
               'Nome da matéria': subjects_df['nome'][subject[0]]}
              for subject in ranking]
    return pd.DataFrame(result, columns=['Relevancia', 'Código da Matéria',
                                         'Nome da matéria'])


def main():
    course = input('Course name: ')
    query = input('Query text: ')
    n = input('Number of similar documents to show (default is 8): ')
    n = int(n) if n else 8
    threshold = input('Minimum similarity threshold value: ')
    threshold = float(threshold) if threshold else 0

    result = search_similarity_query(course, query, n)
    result = result[result['Relevancia'] >= threshold]

    print(result)


if __name__ == '__main__':
    main()
