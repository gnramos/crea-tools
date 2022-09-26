import gensim
import nltk
import pandas as pd
import spacy


class Preprocessor:
    def __init__(self, stopwords, lemmas, noise, nlp):
        self.stopwords = stopwords
        self.lemmas = lemmas
        self.noise = noise
        self.nlp = nlp

    def run(self, text, deacc=True, min_len=2, max_len=46):
        words = gensim.utils.simple_preprocess(text, deacc, min_len, max_len)
        doc = self.nlp(' '.join([w for w in words if w not in self.noise]))
        output = [token.lemma_
                  for token in doc if token.text not in self.stopwords]
        return [self.lemmas.get(lemma, lemma) for lemma in output]

    @staticmethod
    def pt(deacc=True, min_len=2, max_len=46):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        stopwords += ['referente', 'seguinte', 'etc', 'ª', 'tal', 'um', 'dois',
                      'tres', 'vs', 'aula', 'tal']
        words = gensim.utils.simple_preprocess(' '.join(stopwords), deacc=deacc,
                                               min_len=min_len, max_len=max_len)
        stopwords = set(words)
        noise = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi'}
        lemmas = {'campar': 'campo', 'eletromagnetico': 'eletromagnetismo',
                  'matematica': 'matematico', 'Matematica': 'matematico',
                  'Matemática': 'matematico', 'Quimica': 'quimico',
                  'seriar': 'serie'}
        nlp = spacy.load('pt_core_news_lg')
        return Preprocessor(stopwords, lemmas, noise, nlp).run


class Oracle():
    def __init__(self, course_json, preprocess):
        self.preprocess = preprocess

        self.df = pd.read_json(course_json)
        self.df['text'] = self.df[['nome', 'ementa', 'conteudo']].apply(
            lambda x: preprocess('. '.join(x)), axis=1)

        self.nlp = self._make_nlp()

    def _get_similar(self, query):
        vec_bow = self.nlp.id2word.doc2bow(self.preprocess(query))
        vec_tfidf = self.nlp.tfidf[vec_bow]
        vec_lsi = self.nlp.lsi[vec_tfidf]
        return sorted(enumerate(self.nlp.index[vec_lsi]), key=lambda x: x[1],
                      reverse=True)

    def _load_ementa(self, course_json):
        df = pd.read_json(course_json)
        df['text'] = df[['nome', 'ementa', 'conteudo']].apply(
            lambda x: self.preprocess('. '.join(x)), axis=1)
        return df

    def _make_nlp(self):
        from collections import namedtuple

        nlp = namedtuple('NLP', 'id2word, tfidf, lsi, index')
        nlp.id2word = gensim.corpora.Dictionary(self.df['text'])
        corpus = self.df['text'].apply(nlp.id2word.doc2bow).to_list()
        nlp.tfidf = gensim.models.TfidfModel(corpus=corpus, id2word=nlp.id2word)
        nlp.lsi = gensim.models.LsiModel(corpus=nlp.tfidf[corpus],
                                         id2word=nlp.id2word)
        nlp.index = gensim.similarities.MatrixSimilarity(nlp.lsi[nlp.tfidf[corpus]])
        return nlp

    def run(self, query, threshold=0, num_best=5):
        print(query)
        j = 0
        for i, score in self._get_similar(query):
            if score >= threshold:
                print(f'{score:.2f}', self.df.iloc[i]['nome'])
                if (j := j + 1) >= num_best:
                    break
        print()
