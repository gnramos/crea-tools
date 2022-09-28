# crea-tools

Projeto para aplicação de processamento de linguagem natural sobre a documentação de cursos de engenharia. O objetivo é mapear termos específicos à documentação relevante, especificamente termos das definições de competências do [Sistema Confea/Crea](https://www.confea.org.br/sistema-profissional/o-sistema) aos programas de componentes curriculares.

## Preparo

Para utilizar o código, é preciso de algumas ferramentas.

```bash
pip3 install gensim
pip3 install nltk
python3 -m nltk.downloader stopwords
# Spacy tem alguns problemas com a versão do click, melhor atualizar.
pip3 install click --upgrade
pip3 install spacy
python3 -m spacy download pt_core_news_lg
```

A documentação dos cursos (relatórios do [SIGAA](https://sigaa.unb.br/)) dever esta disponível na pasta [data](data) como arquivos HTML. Especificamente:
- o "Relatório da Estrutura Curricular" do _curso_ na pasta [degrees](data/degrees) (acesse [aqui](https://sigaa.unb.br/sigaa/public/curso/lista.jsf?nivel=G) para a listagem de cursos), e
- o "Programa Atual do Componente" curricular, para cada componente na estrutura curricular, na pasta [components](data/components), sendo cada arquivo identificado pelo código da disciplina (acesse [aqui](https://sigaa.unb.br/sigaa/public/componentes/busca_componentes.jsf) para a listagem de componentes).

## Uso

O programa deve ser executado diretamente pela linha de comando:

```bash
python3 process.py {query} {termos}
```

Para mais detalhes, pode-se utilizar o comando:

```bash
python3 process.py -h
```