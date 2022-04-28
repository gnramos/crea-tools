from gensim import corpora
from gensim.similarities.docsim import MatrixSimilarity
from nltk.corpus import stopwords
import gensim
import os
import pandas as pd
import pickle
import spacy

# DataPreprocessing.py ########################################################

def dataPreprocessing(filename:str):
    """ Reads the given file, processes the texts and saves it as a Dataframe into a pickle file.

    This function reads the .json file that contains all subjects and, for each row, concatenates the name, syllabus and content into a single string. This text will then be processed by the preprocess function inserted into a Dataframe. Finally, this Dataframe is saved in a pickle binary text format.

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).

    ### Returns:
        None
    """

    subjects_df = pd.read_json(filename + '.json')
    subjects_df = subjects_df.sort_values(by=["codigo"])
    subjects_df = subjects_df.reset_index(drop=True)

    documents_list = []

    for i, row in subjects_df.iterrows():

        # * reading values of each subject (row)
        subject_id = row["codigo"]
        name = row["nome"]
        syllabus = row["ementa"]
        content = row["conteudo"]

        # * combining them to create the subject document
        text = name + ' ' + syllabus + ' ' + content

        # * preprocessing
        preProcessedText = preprocess(text)
        documents_list.append(preProcessedText)

    documents_series = pd.Series(documents_list, name="documento")

    documents_df = pd.concat([subjects_df, documents_series], axis=1)

    with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'wb') as f:
        pickle.dump(documents_df, f)


# ModelTraining.py ########################################################

def modelTraining(filename:str):
    """ Based on the the given file for the subjects, reads the processed data and trains a tf-idf model and also a lsi-model.

    This function reads the .json file that contains all subjects, which will become the classes for the classification function. Moreover, it will read the saved processed data and train the models. The training consists in, firstly, creating a dictionary that will map all words to an numerical id. Secondly, it will create a bag-of-words corpus that will contain all the words frequency. After that, this values will transformed with the tf-idf model, which will atribute more value to important words that appears less in documents but is very meaningful to its topic. Finally, the values processed by tf-idf will be again transformed using LSI, which performs a SVD (singular value decomposition) and creates a document-to-topic matrix that will saved in a Matrix Market format and be used in the similarity query function. This function also creates a pickle file that will store id2wordDict, tfidf_model, lsi_model objects.

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).

    ### Returns:
        None
    """

    try:
        with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'rb') as f:
             documents_df = pickle.load(f)
    except FileNotFoundError:
        dataPreprocessing(filename=filename)

        with open('AuxFiles\\documents_df_pickle-' + filename + '.txt', 'rb') as f:
            documents_df = pickle.load(f)

    lemmatizedData = documents_df["documento"].tolist()

    # * gensim dictionary object, which will track each word to its respective id
    id2wordDict = corpora.Dictionary(lemmatizedData)

    # * gensim doc2bow method to map each word to a integer id and its respective frequency
    corpus = [id2wordDict.doc2bow(text) for text in lemmatizedData]

    # * corpus -> list of list of tuples (id of a word, frequency)

    tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2wordDict)

    # * num_topics = numbers of subjects
    lsi_model = gensim.models.LsiModel(tfidf_model[corpus], id2word=id2wordDict, num_topics=len(lemmatizedData), power_iters=100)

    gensim.corpora.MmCorpus.serialize('AuxFiles\\tfidf_model-' + filename + '_mm', tfidf_model[corpus])

    gensim.corpora.MmCorpus.serialize('AuxFiles\\lsi_model-' + filename + '_mm', lsi_model[tfidf_model[corpus]])

    with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'wb') as f:
        pickle.dump( [id2wordDict, tfidf_model, lsi_model], f)

# preprocessingFunctionModule.py ##############################################
nlp = spacy.load('pt_core_news_lg')

# * adding custom texts that dont represent real words (noises)
noises_list = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi"]

stopWords_list = stopwords.words("portuguese")

# * adding custom words to StopWords list
stopWords_list += [
    'referente',
    'seguinte',
    'etc',
    'ª',
    'tal',
    'um',
    'dois',
    'tres',
    'vs',
    'aula',
    'tal',
]

# * preprocessing stopwords to correct format
stopWords_list = gensim.utils.simple_preprocess(" ".join(stopWords_list), deacc=True, min_len=1, max_len=40)

# * manual intervention, changing final lemmas
intervention_dict = {
    "campar": "campo",
    "seriar":"serie",
    "eletromagnetico":"eletromagnetismo",
}

def preprocess(text:str):
    """ Preprocesses a given text and returns a list of processed words.

    This function firstly uses the simple_preprocess function from Gensim and removes predefined strings that are considered noises. After that, the function uses the pipeline from Spacy, which has a tokenizer, tagger and parser. Then, it removes stopwords and all words are lemmatized by Spacy. Finally, some predefined lemmas are changed by a dictionary and the remaining lemmas are returned as a list.

    ### Parameters:
        text: a string type object containing the text to be processed.

    ### Returns:
        A list type object containing all words as lemmas.
    """

    # * importing stopwords from nltk and spacy pipeline
    global nlp
    global stopWords_list
    global noises_list
    global intervention_dict

    # * preprocessing text with gensim.simple_preprocess, eliminating noises: lowercase, tokenized, no symbols, no numbers, no accents marks(normatize)
    text_list = gensim.utils.simple_preprocess(text, deacc=True, min_len=1, max_len=40)

    # * recombining tokens to a string type object and removing remaining noises
    text_str = " ".join([word for word in text_list if word not in noises_list])

    # * preprocessing with spacy, retokenizing -> tagging parts of speech (PoS) -> parsing (assigning dependencies between words) -> lemmatizing
    text_doc = nlp(text_str)

    # * re-tokenization, removing stopwords and lemmatizing
    lemmatized_text_list = [token.lemma_ for token in text_doc if token.text not in stopWords_list]

    # * manual intervention conversion of lemmas and removing 1 letter stopwords
    output = []
    for token in lemmatized_text_list:
        if len(token) <= 1:
            continue
        if token in intervention_dict:
            output.append(intervention_dict[token])
        else:
            output.append(token)

    return output

# SimilarityFunction.py ##########################################################
def search_similarity_query(filename:str, search_document:str, num_best:int=8):
    """ Based on the the given text, it will perform a similarity search with the model already trained and return a Dataframe containing the best results.

    Based on the filename given that define the subjects, this function preprocesses the given text and computes the cossine similarity value for each class (subject) that is stored in the model. These values are then sorted in decreacresing order and stored in a Dataframe, which will finally be returned by the function.

    ### Parameters:
        filename: a string type object containing the name of the subjects file (without the extension).
        search_document: a string type object containing the text that will be used in the similarity query.
        num_best: an integer type object which will limit the number of similar subjects returned. By default, this limit will be 8.

    ### Returns:
        A Dataframe type object which will contain the results of the query. This can also be an empty Dataframe if no cossine similarity value is significant enough.
    """

    subjects_df = pd.read_json(filename + '.json')
    subjects_df = subjects_df.sort_values(by=["codigo"])
    subjects_df = subjects_df.reset_index(drop=True)
    try:
        lsi_corpus = gensim.corpora.MmCorpus('AuxFiles\\lsi_model-' + filename + '_mm')
        with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'rb') as f:
            id2wordDict, tfidf_model, lsi_model = pickle.load(f)
    except FileNotFoundError:
        modelTraining(filename=filename)

        lsi_corpus = gensim.corpora.MmCorpus('AuxFiles\\lsi_model-' + filename + '_mm')
        with open('AuxFiles\\dict_and_models_pickle-' + filename + '.txt', 'rb') as f:
            id2wordDict, tfidf_model, lsi_model = pickle.load(f)

    cosineSimilarity = MatrixSimilarity(lsi_corpus, num_features = lsi_corpus.num_terms, num_best=num_best)

    # * preprocessing and processing until becomes a matrix of type term_to_topic (V)
    doc = preprocess(search_document)
    query_bow = id2wordDict.doc2bow(doc)
    query_tfidf = tfidf_model[query_bow]
    query_lsi = lsi_model[query_tfidf]

    # * cossine similarity between the vector of the new document vs all other vectors of documents
    # * returns a list of top num_best tuples (id of compared document, similarity)
    ranking = cosineSimilarity[query_lsi]

    ranking.sort(key=lambda unit: unit[1], reverse= True)
    result = []

    for subject in ranking:

        result.append (
            {
                'Relevancia': round((subject[1] * 100),6),
                'Código da Matéria': subjects_df['codigo'][subject[0]],
                'Nome da matéria': subjects_df['nome'][subject[0]]
            }

        )

    output = pd.DataFrame(result, columns=['Relevancia','Código da Matéria','Nome da matéria'])

    return output

# SimilarityQuery.py ##########################################################
while(1):
    filename = input("Type the name of the file that will provide the subjects (without extension)\n\t")
    if os.path.isfile(filename + '.json'):
        break
    else:
        print('Invalid file, check if the file exists and is typed correctly!\n')
print()

text = input("Type the text that will be used in the query :\n\t")
print()

n = int(input("Type how many most similar documents will be shown:\n\t"))
print()

# TODO: Maybe it's better to use the threshold in a second function. Therefore, 2 functions will be created: one to n-best and other for >= threshold
t = float(input("Would you like to set up a minimum value? [0, 1], type -1 if you don't:\n\t"))
print()

result = search_similarity_query(filename, text, n)

if (t != -1):
    result = result[result['Relevancia'] >= t]

if result.empty:
    print("There were no similar subjects to your inserted text or selected threshold\n")
else:
    print(result)
