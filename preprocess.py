from collections import defaultdict
import os
import pandas as pd
import re


class Course:  # namespace
    @staticmethod
    def read_program(course, encoding='ISO-8859-1', add_bib=False, verbose=False):
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

        d, path = {}, f'data/courses/{course}.html'
        if os.path.isfile(path):
            with open(path, encoding=encoding) as f:
                html = f.read()

            if match := pattern.search(html):
                nome, ementa, bibliografia, conteudo = match.groups()
                d['Código'] = course
                d['Nome'] = nome
                d['Ementa'] = ementa
                d['Conteúdo'] = conteudo
                if add_bib:
                    d['Bibliografia'] = bibliografia
            elif verbose:
                print(f'No matches for {course}.')
        elif verbose:
            print(f'No file "{path}".')

        return d

    @staticmethod
    def program_df(course_list, encoding='ISO-8859-1', add_bib=False, verbose=False):
        """Extrai o programa para cada elemento de uma lista de disciplinas.

        Retorna um DataFrame com as informações.
        """
        d = defaultdict(list)
        for course in course_list:
            for key, value in Course.read_program(course, encoding, add_bib, verbose).items():
                d[key].append(value)

        return pd.DataFrame(d).set_index('Código') if d else None


class Degree:  # namespace
    @staticmethod
    def read_courses(degree, encoding='ISO-8859-1', verbose=False):
        """Extrai a lista de disciplinas de um curso de graduação.

        Assume que existe um arquivo HTML com o conteúdo da página no diretório
        "data/degrees". O nome dever estar no formado NOME.html, onde NOME é o
        nome do curso (ex: mecatronica).

        O arquivo pode se obtido online (após autenticação no SIGAA):
        Ensino > Consultas > Estruturas Curriculares > Estrutura Curricular de Graduação > (Busca) > Relatório da Estrutura Curricular
        """
        componentes = r'componentes">\W+<td>(\w{3}\d{4})</td>\W+<td>\W+(.*?) - (\d+)h[.\s\S]*?<small>\W+(.*?)\W[.\s\S]*?<small>(.*?)</small>'
        cadeias = r'<td>(\w{3}\d{4}) - (.*?) - (\d+)h</td>'
        path = f'data/degrees/{degree}.html'
        courses = set()
        if os.path.isfile(path):
            with open(path, encoding=encoding) as f:
                html = f.read()
            for course, name, hours, course_type, nature in re.findall(componentes, html):
                courses.add(course)
            for course, name, hours in re.findall(cadeias, html):
                courses.add(course)
        elif verbose:
            print(f'No file "{path}".')

        return courses
