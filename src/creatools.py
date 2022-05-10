from gensim.corpora import MmCorpus
from gensim.similarities.docsim import MatrixSimilarity
import gensim
import nltk.corpus
import os
import pandas as pd
import pickle
import spacy
import argparse

def _tmp_dir(file):
    return os.path.join('..', 'tmp', file)


def _get_course_from_json(json_file):
    _, file = os.path.split(json_file)
    course, _ = os.path.splitext(file)
    return course


def _get_subject_df(json_file):
    df = pd.read_json(json_file)
    df = df.sort_values(by=["codigo"])
    return df.reset_index(drop=True)


def model_training(json_file):
    '''Trains tf-idf and lsi models with the data for the course.

    Reads the course's JSON file whose subjects become classes for the training
    the models.

    Keyword arguments:
    json_file -- path the the file with the course data.
    '''

    def document_df():
        course = _get_course_from_json(json_file)
        file = _tmp_dir(f'{course}_documents_df.pkl')
        if os.path.isfile(file):
            return pd.read_pickle(file)

        subjects_df = _get_subject_df(json_file)

        # Provavelmente pode ser feito de modo mais eficiente diretamente no subjects_df
        documents_list = [preprocess(f'{row["nome"]} {row["ementa"]} {row["conteudo"]}')
                          for i, row in subjects_df.iterrows()]
        documents_series = pd.Series(documents_list, name='documento')
        df = pd.concat([subjects_df, documents_series], axis=1)
        df.to_pickle(file)
        return df

    df = document_df()
    lemmatizedData = df["documento"].tolist()
    id2wordDict = gensim.corpora.Dictionary(lemmatizedData)
    corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

    course = _get_course_from_json(json_file)
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
ignore = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi']
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois', 'tres',
              'vs', 'aula', 'tal']
stopwords = gensim.utils.simple_preprocess(' '.join(stopwords), deacc=True,
                                           min_len=1, max_len=40)  # magic numbers?

intervention_dict = {'campar': 'campo',
                     'seriar': 'serie',
                     'eletromagnetico': 'eletromagnetismo'}


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

    def lemmatized_text():
        text_lst = gensim.utils.simple_preprocess(text, deacc=True, min_len=2,
                                                  max_len=46)  # magic numbers?
        text_str = ' '.join([word for word in text_lst if word not in ignore])
        text_doc = nlp(text_str)
        return [token.lemma_
                for token in text_doc
                if token.text not in stopwords]

    return [intervention_dict.get(token, token)
            for token in lemmatized_text()
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
    def get_ranking(course):
        lsi_corpus_file = _tmp_dir(f'{course}_lsi_mm')
        if not os.path.isfile(lsi_corpus_file):
            model_training(json_file)
        lsi_corpus = gensim.corpora.MmCorpus(lsi_corpus_file)

        with open(_tmp_dir(f'{course}_dict+models.pkl'), 'rb') as f:
            id2wordDict, tfidf_model, lsi_model = pickle.load(f)

        cosineSimilarity = MatrixSimilarity(lsi_corpus,
                                            num_features=lsi_corpus.num_terms,
                                            num_best=num_best)

        query_bow = id2wordDict.doc2bow(preprocess(query))
        query_tfidf = tfidf_model[query_bow]
        query_lsi = lsi_model[query_tfidf]

        return sorted(cosineSimilarity[query_lsi],
                      key=lambda unit: unit[1], reverse=True)

    course = _get_course_from_json(json_file)

    df = _get_subject_df(json_file)
    return [(relevance, df['codigo'][idx], df['nome'][idx])
            for idx, relevance in get_ranking(course)
            if relevance >= threshold]


def main():
    parser = argparse.ArgumentParser(description='Search similarity between documents')
    parser.add_argument('-f', '--filename', type=str, metavar='',
                        help="Course JSON file, note: the accepted format is the relative directory and extension. e.g. ../data/mecatronica.json)")
    parser.add_argument('-t', '--text', type=str, metavar='',
                        help="Query text, the text that will be used for searching similar subjects")
    parser.add_argument('-n', '--number', type=int, metavar='',
                        help="Number of similar documents to be shown (default is 8)")
    parser.add_argument('-T', '--threshold', type=float, metavar='',
                        help="Minimum similarity threshold value, an argument to set up a minimum accepted value (default is 0 and has priority over 'number' argument), note: must be a float bewteen 0 and 1 (inclusive)")

    args = parser.parse_args()
    json_file = args.filename
    query = args.text
    n = args.number
    n = int(n) if n else 8
    threshold = args.threshold
    threshold = float(threshold) if threshold else 0

    if not os.path.isfile(args.filename):
        raise argparse.ArgumentTypeError('Invalid file, please check if the file exists and it is inserted correctly')

    result = search_similarity_query(json_file, query, n, threshold)
    idx = 1
    for relevance, code, name in result:
        print(f'{idx:02d} {int(relevance * 100):02d}% {code} {name}')
        idx += 1

if __name__ == '__main__':
    main()