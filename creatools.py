import gensim
import nltk
from preprocess import Component, Degree
import spacy


class Models:  # namespace
    class LsiModel:
        def __init__(self, df):
            self.id2word = gensim.corpora.Dictionary(df)
            corpus = df.apply(self.id2word.doc2bow).to_list()
            self.tfidf = gensim.models.TfidfModel(corpus=corpus, id2word=self.id2word)
            self.lsi = gensim.models.LsiModel(corpus=self.tfidf[corpus], id2word=self.id2word)
            self.index = gensim.similarities.MatrixSimilarity(self.lsi[self.tfidf[corpus]])


class Oracle():
    def __init__(self, degree, nlp_preprocess, nlp_class, verbose=False):
        self.preprocess = nlp_preprocess
        components = Degree.read_components(degree, verbose=verbose)
        df = Component.program_df(components, verbose=verbose)
        self.components_df = df['Nome']
        text_df = df[['Nome', 'Ementa', 'Conteúdo']].apply(
            lambda x: nlp_preprocess('. '.join(x)), axis=1)
        self.nlp = nlp_class(text_df)
        self.verbose = verbose

    def _get_similar(self, query):
        vec_bow = self.nlp.id2word.doc2bow(self.preprocess(query))
        vec_tfidf = self.nlp.tfidf[vec_bow]
        vec_lsi = self.nlp.lsi[vec_tfidf]
        return sorted(enumerate(self.nlp.index[vec_lsi]), key=lambda x: x[1],
                      reverse=True)

    def run(self, query, threshold=0, num_best=5):
        j = 0
        for i, score in self._get_similar(query):
            if score >= threshold:
                print(f'{score:.2f} {self.components_df.iloc[i]}')
                if (j := j + 1) >= num_best:
                    break
            else:
                break
        print()


class NLPPreprocessor:
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
        return NLPPreprocessor(stopwords, lemmas, noise, nlp).run
