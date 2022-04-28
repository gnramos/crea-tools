from gensim.corpora import MmCorpus
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


def model_training(course):
    '''Trains tf-idf and lsi models with the data for the course.

    Reads the course's JSON file whose subjects become classes for the training
    the models.

    Keyword arguments:
    course -- a string containing the name of course.
    '''

    file = f'AuxFiles\\documents_df_pickle-{course}.txt'
    if not os.path.isfile(file):
        dataPreprocessing(course)

    with open(file, 'rb') as f:
        documents_df = pickle.load(f)

    lemmatizedData = documents_df["documento"].tolist()
    id2wordDict = gensim.corpora.Dictionary(lemmatizedData)
    corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

    tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2wordDict)
    file = f'AuxFiles\\tfidf_model-{course}_mm'
    MmCorpus.serialize(file, tfidf_model[corpus])

    lsi_model = gensim.models.LsiModel(tfidf_model[corpus],
                                       id2word=id2wordDict,
                                       num_topics=len(lemmatizedData),
                                       power_iters=100)
    file = f'AuxFiles\\lsi_model-{course}_mm'
    MmCorpus.serialize(file, lsi_model[tfidf_model[corpus]])

    file = f'AuxFiles\\dict_and_models_pickle-{course}.txt'
    with open(file, 'wb') as f:
        pickle.dump([id2wordDict, tfidf_model, lsi_model], f)


nlp = spacy.load('pt_core_news_lg')
ignore = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi"]
stop_words = stopwords.words("portuguese")
stop_words += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois', 'tres',
               'vs', 'aula', 'tal']
stop_words = gensim.utils.simple_preprocess(" ".join(stop_words), deacc=True,
                                            min_len=1, max_len=40)  # magic numbers?

# * manual intervention, changing final lemmas
intervention_dict = {"campar": "campo",
                     "seriar": "serie",
                     "eletromagnetico": "eletromagnetismo"}


def preprocess(text):
    ''' Return a list of processed words from the text.

    Applies Gensim's simple_preprocess, removes words from the ignore_list,
    apply Spacy's the pipeline, remove stopwords, lemmatize words, and remove
    lemmas from the intervention dictionary.

    Keyword arguments:
    text -- string containing the text to be processed.

    Returns:
    A list type object containing all words as lemmas.
    '''

    text_lst = gensim.utils.simple_preprocess(text, deacc=True, min_len=1,
                                              max_len=40)  # magic numbers?
    text_str = " ".join([word for word in text_lst if word not in ignore])
    text_doc = nlp(text_str)
    lemmatized_text = [token.lemma_
                       for token in text_doc
                       if token.text not in stop_words]

    return [intervention_dict.get(token, token)
            for token in lemmatized_text
            if len(token) > 1]


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
        model_training(course)
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
