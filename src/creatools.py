from gensim.corpora import MmCorpus
from gensim.similarities.docsim import MatrixSimilarity
from nltk.corpus import stopwords
import gensim
import os
import pandas as pd
import pickle
import spacy


def _tmp_dir(file):
    return os.path.join('..', 'tmp', file)


def _get_course_from_json(json_file):
    _, file = os.path.split(json_file)
    course, _ = os.path.splitext(file)
    return course


def data_preprocessing(json_file):
    '''Dumps the course information into a pickle file..

    Reads the course's JSON file containing all subjects and builds the
    DataFrame with joined name, syllabus and content.

    Keyword arguments:
    json_file -- path the the file with the course data.
    '''

    subjects_df = pd.read_json(json_file)
    subjects_df = subjects_df.sort_values(by=["codigo"])
    subjects_df = subjects_df.reset_index(drop=True)

    # Provavelmente pode ser feito de modo mais eficiente diretamente no subjects_df
    documents_list = [preprocess(f'{row["nome"]} {row["ementa"]} {row["conteudo"]}')
                      for i, row in subjects_df.iterrows()]
    documents_series = pd.Series(documents_list, name='documento')
    documents_df = pd.concat([subjects_df, documents_series], axis=1)

    course = _get_course_from_json(json_file)
    with open(_tmp_dir(f'{course}_documents_df.pkl'), 'wb') as f:
        pickle.dump(documents_df, f)


def model_training(json_file):
    '''Trains tf-idf and lsi models with the data for the course.

    Reads the course's JSON file whose subjects become classes for the training
    the models.

    Keyword arguments:
    json_file -- path the the file with the course data.
    '''

    course = _get_course_from_json(json_file)
    file = _tmp_dir(f'{course}_documents_df.pkl')
    if not os.path.isfile(file):
        data_preprocessing(json_file)

    with open(file, 'rb') as f:
        documents_df = pickle.load(f)

    lemmatizedData = documents_df["documento"].tolist()
    id2wordDict = gensim.corpora.Dictionary(lemmatizedData)
    corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

    tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2wordDict)
    MmCorpus.serialize(_tmp_dir(f'{course}_tfidf_mm'), tfidf_model[corpus])

    lsi_model = gensim.models.LsiModel(tfidf_model[corpus],
                                       id2word=id2wordDict,
                                       num_topics=len(lemmatizedData),
                                       power_iters=100)
    MmCorpus.serialize(_tmp_dir(f'{course}_lsi_mm'),
                       lsi_model[tfidf_model[corpus]])

    with open(_tmp_dir(f'{course}_dict+models.pkl'), 'wb') as f:
        pickle.dump([id2wordDict, tfidf_model, lsi_model], f)


nlp = spacy.load('pt_core_news_lg')
ignore = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi"]
stop_words = stopwords.words("portuguese")
stop_words += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois', 'tres',
               'vs', 'aula', 'tal']
stop_words = gensim.utils.simple_preprocess(" ".join(stop_words), deacc=True,
                                            min_len=1, max_len=40)  # magic numbers?

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


def search_similarity_query(json_file, query, num_best=8, threshold=0):
    '''Apply the course model to search for query within subjects.

    Keyword arguments:
    json_file -- path the the file with the course data.
    query -- text to be searched for.
    num_best -- the number of similar subjects shown.
    threshold -- the minimum relevance threshold to consider.

    Returns:
        A list of tuples with the resulting disciplines in the format
        (relevance, code, name).
    '''

    subjects_df = pd.read_json(json_file)
    subjects_df = subjects_df.sort_values(by=['codigo'])
    subjects_df = subjects_df.reset_index(drop=True)

    course = _get_course_from_json(json_file)
    lsi_corpus_file = _tmp_dir(f'{course}_lsi_mm')
    if not os.path.isfile(lsi_corpus_file):
        model_training(json_file)
    lsi_corpus = gensim.corpora.MmCorpus(lsi_corpus_file)

    with open(_tmp_dir(f'{course}_dict+models.pkl'), 'rb') as f:
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
    return [(relevance, subjects_df['codigo'][idx], subjects_df['nome'][idx])
            for idx, relevance in ranking
            if relevance >= threshold]


def main():
    json_file = input('Course JSON file: ')
    query = input('Query text: ')
    n = input('Number of similar documents to show (default is 8): ')
    n = int(n) if n else 8
    threshold = input('Minimum similarity threshold value: ')
    threshold = float(threshold) if threshold else 0

    result = search_similarity_query(json_file, query, n, threshold)
    for relevance, code, name in result:
        print(f'{int(relevance * 100):3d}% {code} {name}')


if __name__ == '__main__':
    main()
