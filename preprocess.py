from collections import defaultdict
import os
import pandas as pd
import re


class Component:  # namespace
    @staticmethod
    def read_program(component, encoding='ISO-8859-1', add_bib=False, verbose=False):
        """Extrai o programa da página HTML da disciplina.

        Assume que existe um arquivo HTML com o conteúdo da página no diretório
        "data/courses". O nome dever estar no formado COD.html, onde COD é o
        código da disciplina (ex: CIC0004).

        O arquivo pode se obtido online (após autenticação no SIGAA):
        Ensino > Consultas > Componentes Curriculares > (Busca) > Programa atual do componente
        """
        pattern = re.compile(r'Componente Curricular:</th>[.\s\S]*?<td>\w{3}\d{4}\W*(.*?)</td>[.\s\S]*?'
                             r'Ementa:</th>[\W]*?<td>\W*(.*?)\W*</td>[.\s\S]*?'
                             r'Objetivos:</th>[.\s\S]*?itemPrograma">\W*(.*?)\W*</td>[.\s\S]*?'
                             r'Conteúdo:</th>[.\s\S]*?itemPrograma">\W*(.*?)\W*</td>')

        d, path = {}, f'data/courses/{component}.html'
        if os.path.isfile(path):
            with open(path, encoding=encoding) as f:
                html = f.read()

            if match := pattern.search(html):
                d['Código'] = component
                d['Nome'], d['Ementa'], d['Conteúdo'], bib = match.groups()
                if add_bib:
                    d['Bibliografia'] = bib
            elif verbose:
                print(f'No matches for {component}.')
        elif verbose:
            print(f'No file "{path}".')

        return d

    @staticmethod
    def program_df(course_list, encoding='ISO-8859-1', add_bib=False, verbose=False):
        """Extrai o programa para cada elemento de uma lista de disciplinas.

        Retorna um DataFrame com as informações.
        """
        d = defaultdict(list)
        for component in course_list:
            for key, value in Component.read_program(component, encoding, add_bib, verbose).items():
                d[key].append(value)

        return pd.DataFrame(d).set_index('Código')


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
