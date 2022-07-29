from gensim.utils import simple_preprocess
import argparse
import gensim
import nltk
import pandas as pd
import spacy

# Variáveis globais
stopwords = noises = lemma_intervention = nlp = None


def _init_global():
    global stopwords, noises, lemma_intervention, nlp
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois',
                  'tres', 'vs', 'aula', 'tal']
    stopwords = set(simple_preprocess(' '.join(stopwords), deacc=True,
                                      min_len=1, max_len=46))
    noises = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi'}
    lemma_intervention = {'campar': 'campo',
                          'seriar': 'serie',
                          'eletromagnetico': 'eletromagnetismo',
                          'Quimica': 'quimico',
                          'Matematica': 'matematico'}
    nlp = spacy.load('pt_core_news_lg')


def _preprocess(text, deacc=True, min_len=2, max_len=46):
    doc = nlp(' '.join([word for word in simple_preprocess(text, deacc,
                                                           min_len, max_len)
                        if word not in noises]))
    output = [token.lemma_ for token in doc if token.text not in stopwords]
    return [lemma_intervention.get(lemma, lemma) for lemma in output]


def _get_df_id2word(json_file):
    df = pd.read_json(json_file)
    df['text'] = df[['nome', 'ementa', 'conteudo']].apply(
        lambda x: _preprocess('. '.join(x)), axis=1)
    id2word = gensim.corpora.Dictionary(df['text'])
    df['corpus'] = df['text'].apply(id2word.doc2bow)
    return df, id2word


def _get_similar(query, id2word, tfidf, lsi, index):
    vec_bow = id2word.doc2bow(_preprocess(query))
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    return sorted(enumerate(index[vec_lsi]), key=lambda x: x[1], reverse=True)


def _parse_args():
    def check_positive(n):
        n = int(n)
        if n <= 0:
            raise argparse.ArgumentTypeError(f'n={n} <= 0')
        return n

    def check_threshold(t):
        t = float(t)
        if not (-1.0 <= t <= 1.0):
            raise argparse.ArgumentTypeError(f't={t} ∉ [-1.0, 1.0]')
        return t

    parser = argparse.ArgumentParser(description='Busca por similaridade')
    parser.add_argument('course', help='nome do curso com as informações')
    parser.add_argument('-n', '--num_best', type=check_positive, default=5,
                        help='número máximo de tópicos a retornar')
    parser.add_argument('-t', '--threshold', type=check_threshold, default=0.0,
                        help='valor mínimo de similaridade aceito [-1, 1]')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--query', nargs='+',
                       help='query única')
    group.add_argument('-m', '--multi_query', action='store_true',
                       help='múltiplas queries')

    return parser.parse_args()


def main():
    def _show(query):
        j, similar = 0, _get_similar(query, id2word, tfidf, lsi, index)
        for i, score in similar:
            if score >= args.threshold:
                print(f'{score:.2f}', df.iloc[i]['nome'])
                j += 1
            if j >= args.num_best:
                break

    args = _parse_args()

    _init_global()
    json_file = f'cursos/{args.course}.json'
    df, id2word = _get_df_id2word(json_file)
    corpus = df['corpus'].to_list()
    tfidf = gensim.models.TfidfModel(corpus=corpus, id2word=id2word)
    lsi = gensim.models.LsiModel(corpus=tfidf[corpus], id2word=id2word)
    index = gensim.similarities.MatrixSimilarity(lsi[tfidf[corpus]])

    if args.multi_query:
        while query := input('Type new query (empty to stop): '):
            _show(query)
    else:
        _show(' '.join(args.query))


if __name__ == '__main__':
    main()
