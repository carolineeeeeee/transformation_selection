from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re
import textacy
from nltk.stem import PorterStemmer
porter = PorterStemmer()

# need to find the maximum match
def find_keywords(text, stemmed_keywords): 
    # stem everything, loop over keywords to find matches
    token_words=word_tokenize(text)
    stemmed = [porter.stem(word) for word in token_words]
    clean_text = " ".join(stemmed)    
    # TODO: find word match, not just letter match
    found_keywords = [w for w in stemmed_keywords.keys() if any([(syn in clean_text) for syn in stemmed_keywords[w]])]#.split()]
    # TODO: group them some how
    # this will give a big list of words that may have duplicates
    return found_keywords

def stem(text):
	token_words=word_tokenize(text)
	stemmed = [porter.stem(word) for word in token_words]
	clean_text = " ".join(stemmed)
	return clean_text

# load image processing keywords
def load_keywords(keywords_file):
    f = open(keywords_file, "r")
    keywords = {}
    
    for wordline in f.readlines():
        abbrv = None
        if '#' in wordline: 
            continue
        if len(wordline.strip()) == 0:
            continue
        if (wordline.strip()).isupper(): # if its an abbreviation
            keywords[(wordline.strip()).lower()] = wordline.strip()
            continue
        if '(' in wordline:
            words = re.findall(r'([^\(\)]*)\s+\(([^\(\)]*)\)', wordline.strip())
            wordline = words[0][0]
            abbrv = words[0][1]
        word = wordline.split(':')[0]
        clean_words = stem(word)
        if clean_words not in keywords.keys():
            w = wordline.strip()
            keywords[w] = []
            keywords[w].append(clean_words)
            synonyms = list(set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets(w)])))
            synonyms = [porter.stem(s) for s in synonyms] # including the original word
            if abbrv:
                synonyms.append(abbrv)
            keywords[w] += synonyms
    return keywords

def group_keyword_results(entry_keywords_to_transf): # entry_keywords_to_transf[entry.risk_id][word].append(t.name)
    word_groups = {}
    return 0

def check_parcing(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    for np in doc.noun_chunks:
        print(np.text)
