from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re
import textacy
from nltk.stem import PorterStemmer
porter = PorterStemmer()


def find_keywords(texts, stemmed_keywords, is_transf=False): 
    # stem everything, loop over keywords to find matches
    

    if is_transf:
        results = []
        for text in texts:
            token_words=word_tokenize(text)
            stemmed = {porter.stem(word) for word in token_words}
            results += list([w for w in stemmed_keywords.keys() if set(stemmed) >= set(word_tokenize(stemmed_keywords[w][0]))])
            to_remove = []
        for w in list(set(results)):
            if ' ' in w:
                splitted = w.split()
                #if 'image' not in splitted:
                to_remove += splitted
        return [w for w in results if w not in to_remove] 
    else:
        all_results = []
        for l in texts:
            results = []
            for text in l:
                token_words=word_tokenize(text)
                stemmed = {porter.stem(word) for word in token_words}
                results += list([w for w in stemmed_keywords.keys() if any([set(stemmed) >= set(word_tokenize(s)) for s in stemmed_keywords[w]])])
            to_remove = []
            for w in list(set(results)):
                if ' ' in w:
                    splitted = w.split()
                    #if 'image' not in splitted:
                    to_remove += splitted
            all_results.append([w for w in results if w not in to_remove] )
    return all_results
    '''
    token_words=word_tokenize(text)
    stemmed = [porter.stem(word) for word in token_words]
    clean_text = ' ' + " ".join(stemmed) + ' '    
    found_keywords = [w for w in stemmed_keywords.keys() if any([(' ' + syn+' ' in clean_text) for syn in stemmed_keywords[w]])]#.split()]
    # if light source exist, remove light and source
    to_remove = []
    for w in found_keywords:
        if ' ' in w:
            to_remove += w.split()
        
    return [w for w in found_keywords if w not in to_remove] 
    '''

def group_keywords(entry_keywords_to_transf, stemmed_keywords):
    grouped_match = {}
    for entry in entry_keywords_to_transf:
        if entry not in grouped_match:
            grouped_match[entry] = {}
        list_keywords = entry_keywords_to_transf[entry].keys()
        #print(list_keywords)
        groups = []
        for k in list_keywords:
            if k not in sum(groups, []):
                similar = [j for j in list_keywords if porter.stem(j) in stemmed_keywords[k]]
                groups.append(similar)
        #print(groups)
        
        # TODO: remove words like image and scene (might be okay if we consider all entries and remove repeated keywords)
        # temp fix: manually remove
        for group in groups:
            group_name = ', '.join(group)
            if 'image' in group_name or 'scene' in group_name: #temp fix
                continue 
            group_match = list(set(sum([entry_keywords_to_transf[entry][k] for k in group], [])))
            grouped_match[entry][group_name] = group_match
    print(grouped_match)
    return grouped_match

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
            synonyms = [porter.stem(s) for s in synonyms if 'random' not in s] # including the original word
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

