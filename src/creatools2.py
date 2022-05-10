import argparse
from gensim import similarities
from gensim.utils import simple_preprocess
import gensim
import nltk
import os
import pandas as pd

stopwords = None


def _preprocess(text, deacc=True, min_len=2, max_len=40):
    global stopwords
    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('portuguese')
    return [token for token in simple_preprocess(text, deacc, min_len, max_len)
            if token not in stopwords]


def _get_df_id2word(course, json_file):
    file = os.path.join('..', 'tmp', f'{course}.corpus')
    update_files = not os.path.isfile(file)
    if update_files:
        df = pd.read_json(json_file)
        df['text'] = df[['nome', 'ementa', 'conteudo']].apply(lambda x: _preprocess('. '.join(x)), axis=1)
        id2word = gensim.corpora.Dictionary(df['text'])
        df['corpus'] = df['text'].apply(id2word.doc2bow)
        df[['corpus', 'text', 'nome']].to_pickle(file)
    else:
        df = pd.read_pickle(file)
        id2word = gensim.corpora.Dictionary(df['text'])

    return df, id2word


def _get_index(course, corpus, lsi):
    file = os.path.join('..', 'tmp', f'{course}.index')
    if os.path.isfile(file):
        index = similarities.MatrixSimilarity.load(file)
    else:
        index = similarities.MatrixSimilarity(lsi[corpus])
        index.save(file)
    return index


def _get_lsi(course, corpus, id2word):
    file = os.path.join('..', 'tmp', f'{course}.lsi')
    if os.path.isfile(file):
        lsi = gensim.models.LsiModel.load(file)
    else:
        lsi = gensim.models.LsiModel(corpus, id2word=id2word)  #, num_topics=2)
        lsi.save(file)
    return lsi


def _get_similar(query, id2word, lsi, index):
    vec_bow = id2word.doc2bow(_preprocess(query))
    vec_lsi = lsi[vec_bow]
    return sorted(enumerate(index[vec_lsi]), key=lambda x: x[1], reverse=True)


def _parse_args():
    parser = argparse.ArgumentParser(description='query')
    parser.add_argument('course', help='nome do curso com as informações')
    parser.add_argument('query', nargs='+',
                        help='palavras a buscar nas informações do curso')
    parser.add_argument('-n', '--num_best', type=int, default=5,
                        help='numero máximo de tópicos a retornar')

    return parser.parse_args()


def main():
    args = _parse_args()
    json_file = f'../data/{args.course}.json'
    df, id2word = _get_df_id2word(args.course, json_file)
    lsi = _get_lsi(args.course, df['corpus'], id2word)
    index = _get_index(args.course, df['corpus'], lsi)
    similar = _get_similar(' '.join(args.query), id2word, lsi, index)
    for i, score in similar[:args.num_best]:
        print(f'{score:.2f}', df.iloc[i]['nome'])


if __name__ == '__main__':
    main()
