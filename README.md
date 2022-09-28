# crea-tools

Projeto para aplicação de processamento de linguagem natural sobre a documentação de cursos de engenharia.

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
- o programa de cada componente curricular na pasta [courses](data/courses), sendo cada arquivo identificado pelo código da disciplina. (acesse [aqui](https://sigaa.unb.br/sigaa/public/componentes/busca_componentes.jsf) para a listagem de cursos).

## Uso

O programa deve ser executado diretamente pela linha de comando, basta utilizar o seguinte comando:

```bash
python3 process.py {curso} -q {query}
```

Para mais detalhes, pode-se utilizar o comando:

```bash
python3 process.py -h
```
## Disclaimer:

Para saber quais os conteúdos exigidos para cada competência e quais as atribuições de cada curso, utilizou-se como referencial uma planilha obtida a partir de um ex-conselheiro do CREA. Essa planilha não é oficial, mas estava sendo discutido pelo CREA a validade desse documento.

