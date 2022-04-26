import spacy
import gensim
from nltk.corpus import stopwords

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