import argparse
import gensim
import nltk
import os
import pandas as pd
import spacy
 
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois', 'tres', 'vs', 'aula', 'tal']
# inclui-se todos os stopwords (len in [1, 46]) para que a função _preprocess consiga corretamente definir o tamanho desejado das palavras
stopwords = set(gensim.utils.simple_preprocess(" ".join(stopwords), deacc=True, min_len=1, max_len=46))
noises = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi"}
lemma_intervention = {
    "campar": "campo",
    "seriar":"serie",
    "eletromagnetico":"eletromagnetismo",
    "Quimica": "quimico",
    "Matematica": "matematico",
}
nlp = spacy.load('pt_core_news_lg')

def _preprocess(text, deacc=True, min_len=2, max_len=46):
    global stopwords, noises, lemma_intervention, nlp
    doc = nlp(' '.join([word for word in gensim.utils.simple_preprocess(text, deacc, min_len, max_len)
                        if word not in noises]))
    output = [token.lemma_ for token in doc if token.text not in stopwords]
    return [lemma_intervention[lemma] if lemma in lemma_intervention else lemma for lemma in output]

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
    parser = argparse.ArgumentParser(description='similarity query')
    parser.add_argument('course', help='nome do curso com as informações')
    parser.add_argument('query', nargs='+',
                        help='palavras a buscar nas informações do curso')
    parser.add_argument('-n', '--num_best', type=int, default=5,
                        help='número máximo de tópicos a retornar')
    parser.add_argument('-t', '--threshold', type=float, default=0.0,
                        help='número mínimo de similaridade aceito')
    return parser.parse_args()

def main():
    args = _parse_args()
    json_file = f'../data/{args.course}.json'
    df, id2word = _get_df_id2word(json_file)
    corpus = df['corpus'].to_list()
    tfidf = gensim.models.TfidfModel(corpus=corpus, id2word=id2word)
    lsi = gensim.models.LsiModel(corpus=tfidf[corpus], id2word=id2word)
    index = gensim.similarities.MatrixSimilarity(lsi[tfidf[corpus]])
    similar = _get_similar(' '.join(args.query), id2word, tfidf, lsi, index)
    for i, score in similar[:args.num_best]:
        if score >= args.threshold:
            print(f'{score:.2f}', df.iloc[i]['nome'])

if __name__ == '__main__':
    main()
