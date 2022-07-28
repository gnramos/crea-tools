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

## Uso

O programa deve ser executado diretamente pela linha de comando, basta utilizar o seguinte comando:

```bash
python3 creatools.py {curso} {query}
```

Para mais detalhes, pode-se utilizar o comando:

```bash
python3 creatools.py -h
```
## Disclaimer:

Para saber quais os conteúdos exigidos para cada competência e quais as atribuições de cada curso, utilizou-se como referencial uma planilha obtida a partir de um ex-conselheiro do CREA. Essa planilha não é oficial, mas estava sendo discutido pelo CREA a validade desse documento.

