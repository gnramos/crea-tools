from collections import defaultdict
import gensim
import nltk
import os
import pandas as pd
import re
import spacy


class Models:  # namespace
    class LsiModel:
        def __init__(self, df):
            self.id2word = gensim.corpora.Dictionary(df)
            corpus = df.apply(self.id2word.doc2bow).to_list()
            self.tfidf = gensim.models.TfidfModel(corpus=corpus, id2word=self.id2word)
            self.lsi = gensim.models.LsiModel(corpus=self.tfidf[corpus], id2word=self.id2word)
            self.index = gensim.similarities.MatrixSimilarity(self.lsi[self.tfidf[corpus]])


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


class Oracle():
    def __init__(self, degree, nlp_preprocess, nlp_class, verbose=False):
        self.preprocess = nlp_preprocess
        components = SIGAA.Degree.read_components(degree, verbose=verbose)
        df = SIGAA.Component.program_df(components, verbose=verbose)
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


class SIGAA:  # namespace
    class Component:  # namespace
        @staticmethod
        def read_program(component, encoding='ISO-8859-1', add_bib=False, verbose=False):
            """Extrai o programa da página HTML da disciplina.

            Assume que existe um arquivo HTML com o conteúdo da página no diretório
            "data/components". O nome dever estar no formado COD.html, onde COD é o
            código da disciplina (ex: CIC0004).

            O arquivo pode se obtido online (após autenticação no SIGAA):
            Ensino > Consultas > Componentes Curriculares > (Busca) > Programa atual do componente
            """
            pattern = re.compile(r'Componente Curricular:</th>[.\s\S]*?<td>\w{3}\d{4}\W*(.*?)</td>[.\s\S]*?'
                                 r'Ementa:</th>[\W]*?<td>\W*(.*?)\W*</td>[.\s\S]*?'
                                 r'Objetivos:</th>[.\s\S]*?itemPrograma">\W*(.*?)\W*</td>[.\s\S]*?'
                                 r'Conteúdo:</th>[.\s\S]*?itemPrograma">\W*(.*?)\W*</td>')

            program, path = {}, f'data/components/{component}.html'
            if os.path.isfile(path):
                with open(path, encoding=encoding) as f:
                    html = f.read()

                if match := pattern.search(html):
                    program['Código'] = component
                    program['Nome'], program['Ementa'], program['Conteúdo'], bib = match.groups()
                    if add_bib:
                        program['Bibliografia'] = bib
                elif verbose:
                    print(f'No matches for {component}.')
            elif verbose:
                print(f'No file "{path}".')

            return program

        @staticmethod
        def program_df(course_list, encoding='ISO-8859-1', add_bib=False, verbose=False):
            """Extrai o programa para cada elemento de uma lista de disciplinas.

            Retorna um DataFrame com as informações.
            """
            program = defaultdict(list)
            for component in course_list:
                for key, value in SIGAA.Component.read_program(component, encoding, add_bib, verbose).items():
                    program[key].append(value)

            return pd.DataFrame(program).set_index('Código')

    class Degree:  # namespace
        @staticmethod
        def read_components(degree, encoding='ISO-8859-1', verbose=False):
            """Extrai a lista de disciplinas de um curso de graduação.

            Assume que existe um arquivo HTML com o conteúdo da página no diretório
            "data/degrees". O nome dever estar no formado NOME.html, onde NOME é o
            nome do curso (ex: mecatronica).

            O arquivo pode se obtido online (após autenticação no SIGAA):
            Ensino > Consultas > Estruturas Curriculares > Estrutura Curricular de Graduação > (Busca) > Relatório da Estrutura Curricular
            """
            # component: code, name, hours, type, nature
            comp_re = r'componentes">\W+<td>(\w{3}\d{4})</td>\W+<td>\W+(.*?) - (\d+)h[.\s\S]*?<small>\W+(.*?)\W[.\s\S]*?<small>(.*?)</small>'
            # chain: code, name, hours
            chain_re = r'<td>(\w{3}\d{4}) - (.*?) - (\d+)h</td>'
            path = f'data/degrees/{degree}.html'
            components = set()
            if os.path.isfile(path):
                with open(path, encoding=encoding) as f:
                    html = f.read()
                components.update(match[0] for match in re.findall(comp_re, html))
                components.update(match[0] for match in re.findall(chain_re, html))
            elif verbose:
                print(f'No file "{path}".')

            return components
