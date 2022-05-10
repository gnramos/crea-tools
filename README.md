# crea-tools

Projeto para aplicação de processamento de linguagem natural sobre a documentação de cursos de engenharia.

## Setup

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