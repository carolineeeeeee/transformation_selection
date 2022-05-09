from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer
porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
lancaster = LancasterStemmer()
def reduce(x):
    if 'blurry' in x:
        words = x.split()
        results = ''
        for w in words:
            if w == 'blurry':
                results += 'blur '
            else:
                results += wordnet_lemmatizer.lemmatize(w)
        return results[:-1]
    else:
        return wordnet_lemmatizer.lemmatize(x)

def new_reduce(x):
    result = []
    for w in x:
        if w == 'blurry':
            result.append('blur')
        else:
            result.append(wordnet_lemmatizer.lemmatize(w))
    return result

def find_keywords(texts, stemmed_keywords, is_transf=False): 
    # stem everything, loop over keywords to find matches
    if is_transf: # looking for exact match
        found_keywords = {}
        for text in texts:
            token_words=word_tokenize(text)
            stemmed = {reduce(word) for word in token_words}
            for w in stemmed_keywords.keys():
                if set(stemmed) >= set(word_tokenize(reduce(w))):
                    if len(word_tokenize(reduce(w))) > 0:
                        found_keywords[w] = word_tokenize(reduce(w))
        to_remove = []
        for w in found_keywords:
            if any([set(found_keywords[x])>set(found_keywords[w]) for x in found_keywords]):
                to_remove.append(w)
        return [w for w in found_keywords.keys() if w not in to_remove] 
    else: # also looking for synonyms
        all_results = []
        found_keywords = {}
        for l in texts:
            found_keywords_l = {}
            for text in l:
                token_words=word_tokenize(text)
                stemmed = {reduce(word) for word in token_words}
                for w in stemmed_keywords.keys():
                    for s in stemmed_keywords[w]:
                        if set(stemmed) >= set(word_tokenize(s)):
                            if len(word_tokenize(s)) > 0:
                                found_keywords_l[w] = word_tokenize(s)
                                break
            to_remove = []
            for w in found_keywords_l:
                if 'image' in w:
                    continue
                if any([set(found_keywords_l[x])>set(found_keywords_l[w]) for x in found_keywords_l]):
                    to_remove.append(w)
            all_results.append([w for w in found_keywords_l if w not in to_remove])
            found_keywords.update(found_keywords_l)


        return all_results, found_keywords

def group_keywords(entry_keywords_to_transf, entry):
    # group based on match
    groups = []
    list_keywords = entry_keywords_to_transf[entry.risk_id]
    for k in list_keywords:
        if k not in sum(groups, []):
            same_match = [j for j in list_keywords if set(entry.found_keywords[j]) == set(entry.found_keywords[k])]
            groups.append(same_match)
    grouped_match = {}
    
    for group in groups:
        if 'object' in group or 'sensor' in group or 'region' in group or 'pixel' in group:
            continue
        group_name = ', '.join(group)
        if 'image' in group_name or 'scene' in group_name: #temp fix
            continue 
        group_match = []
        for k in group:
            group_match += entry_keywords_to_transf[entry.risk_id][k]
        group_match = list(set(group_match))
        grouped_match[group_name] = group_match

    return grouped_match

def group_keywords_2(entry_keywords_to_transf, stemmed_keywords):
    grouped_match = {}
    for entry in entry_keywords_to_transf:
        if entry not in grouped_match:
            grouped_match[entry] = {}
        list_keywords = entry_keywords_to_transf[entry].keys()
        groups = []
        for k in list_keywords:
            if k not in sum(groups, []):
                similar = [j for j in list_keywords if reduce(j) in stemmed_keywords[k]]
                groups.append(similar)

        for group in groups:
            group_name = ', '.join(group)
            if 'image' in group_name or 'scene' in group_name: #temp fix
                continue 
            group_match = list(set(sum([entry_keywords_to_transf[entry][k] for k in group], [])))
            grouped_match[entry][group_name] = group_match
    return grouped_match

def stem(text):
	token_words=word_tokenize(text)
	stemmed = [reduce(word) for word in token_words]
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
            synonyms = [reduce(s) for s in synonyms if 'random' not in s] # including the original word
            if abbrv:
                synonyms.append(abbrv)
            keywords[w] += synonyms
    return keywords

def check_parcing(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    for np in doc.noun_chunks:
        print(np.text)
