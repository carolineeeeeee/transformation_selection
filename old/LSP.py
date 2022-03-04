from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re
import textacy
from transformations_matching import * 

'''

class LS_Pattern:
    
    parser = ''
    type = 1 # look inside (0) np or look before (1) or look after (2)
    regex = '(too .*)'
    LSP_pattern = ''
    nlp = spacy.load("en_core_web_lg")

    def __init__(self, regex, LSP_pattern, parser="en_core_web_lg"):
        self.LSP_pattern = LSP_pattern
        self.regex = regex
        self.parser = parser
    
    def parce(self, text): # too faint light / light is too faint
        doc = self.nlp(text)
        #doc = self.nlp("too faint light in (part of) the scene")
        results = []
        # remove in the scene, part of the scene etc.
        # removing ADP + NP after that

        # np followed by punctuation or end
        text_blocks = re.findall(r"([\w'\s]+)|[.,!?;()''\"\"]", text)
        for block in text_blocks:
            if block != '' and block.strip() in [np.text for np in doc.noun_chunks]:
                results.append(block.strip())
        
        for i in range(len(doc)):
            # too ADJ NOUN, e.g., too faint light
            #if doc[i].lemma_ == 'too' and i < len(doc)-2:
            if doc[i].pos_ == 'ADV' and i < len(doc)-2 and ('N' in doc[i+2].pos_ or (doc[i+2].pos_=='ADJ' and doc[i+2].lemma_ == 'light')) : # a calibration because of the mistake of the parser
                #if doc[i+1].pos_ == 'ADJ':
                #    text_part = doc[i].lemma_ + ' ' 
                #    for j in range(i+2, len(doc)):
                #        text_part+= doc[j].lemma_ + ' '
                    #
                #        if 'N' in doc[j].pos_:                           
                results.append(doc[i].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)
                #break
            # NOUN ... is too ADJ, e.g., light is too faint
            if doc[i].pos_ == 'AUX' and i < len(doc)-2:
                if doc[i+1].pos_ == 'ADV' and doc[i+2].pos_ == 'ADJ':
                    # find the noun right before them
                    for j in range(i+1):
                        if doc[i-j].pos_ == 'NOUN':# and doc[i-j].text != 'scene':
                            results.append(doc[i-j].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)
                            #break
            #if doc[i].pos_ == 'PUNCT' and i < len(doc)-2:
            #    if doc[i+1].pos_ == 'NOUN' and doc[i+2].pos_ == 'PUNCT':
            #        results.append(doc[i+1].lemma_)

        return set(results)
        for token in doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop)
        for np in doc.noun_chunks:
            print(np.text)
        #print(text)
        #regex_match = re.findall(self.regex, text)
        #print(regex_match)
        #if len(regex_match) == 0:
        #    return []
        #doc = textacy.make_spacy_doc(regex_match[0][0], lang='en_core_web_sm')
        #lists = textacy.extract.matches.token_matches(doc, self.LSP_pattern)
        #return [x.text for x in lists]
'''

nlp = spacy.load("en_core_web_lg")

def parce(text): # too faint light / light is too faint
    doc = nlp(text)
    #doc = self.nlp("too faint light in (part of) the scene")
    results = []
    # remove in the scene, part of the scene etc.
    # removing ADP + NP after that

    # np followed by punctuation or end
    #results += parce_np(text, doc)
        
    # TODO: remove punctuations
    no_punct = []
    for token in doc:
        if token.pos_ != 'PUNCT':
            no_punct.append(token)
    # too much, very much, something is too much etc.
    results += parce_too(no_punct)

    # different than expected
    #results += parce_different(doc)

    return set(results)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    for np in doc.noun_chunks:
        print(np.text)
    #print(text)
    #regex_match = re.findall(self.regex, text)
    #print(regex_match)
    #if len(regex_match) == 0:
    #    return []
    #doc = textacy.make_spacy_doc(regex_match[0][0], lang='en_core_web_sm')
    #lists = textacy.extract.matches.token_matches(doc, self.LSP_pattern)
    #return [x.text for x in lists]

def parce_np(text, doc):
    #doc = self.nlp("too faint light in (part of) the scene")
    results = []
    # remove in the scene, part of the scene etc.
    # removing ADP + NP after that

    # np followed by punctuation or end
    text_blocks = re.findall(r"([\w'\s]+)|[.,!?;()''\"\"]", text)
    for block in text_blocks:
        if block != '' and block.strip() in [np.text for np in doc.noun_chunks]:
            results.append(block.strip())
    return results

def parce_vp():
    return 0

def parce_can_be():
    return 0

def parce_more_less(doc): # patterns for keyword 'more' and 'less'
    results = []
    # TODO: too/very
    for i in range(len(doc)):
        # too ADJ NOUN, e.g., too faint light
        if doc[i].pos_ == 'ADV' and (doc[i].text == 'too' or doc[i].text == 'very'):  
            if i < len(doc)-2 and ('N' in doc[i+2].pos_ or (doc[i+2].pos_=='ADJ' and doc[i+2].lemma_ == 'light')) : # a calibration because of the mistake of the parser                        
                results.append(doc[i].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)

        # NOUN ... is too ADJ, e.g., light is too faint
        if doc[i].pos_ == 'AUX' and i < len(doc)-2:
            if doc[i+1].pos_ == 'ADV' and (doc[i+1].text == 'too' or doc[i+1].text == 'very'): 
                if doc[i+2].pos_ == 'ADJ':
                    # find the noun right before them
                    for j in range(i+1):
                        if doc[i-j].pos_ == 'NOUN':# and doc[i-j].text != 'scene':
                            results.append(doc[i-j].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)    
    # NP is more ADJ or ADJ than expected

    return results

def parce_different(doc):
    results = []
    for i in range(len(doc)):
    # np is different than/from expected: ADV/ADJ SCONJ VERB/ADJ
        if (doc[i].pos_ == 'ADV' or doc[i].pos_ == 'ADJ') and i < len(doc)-2:
            if (doc[i+1].pos_ == 'SCONJ' and doc[i+2].pos_ == 'VERB') or (doc[i+1].pos_ == 'ADP' and doc[i+2].pos_ == 'ADJ'):
                results.append(doc[i].text + doc[i+1].text + doc[i+2].text)
        # np has a different np than/from expected

        # other than
    return results

def check_parcing(text):
    doc = nlp(text)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    for np in doc.noun_chunks:
        print(np.text)

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
            words = re.findall(r'([\w\s]+) \((\w+)\)', wordline.strip())
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


# need to find the maximum match
def find_keywords(text, stemmed_keywords): 
    # stem everything, loop over keywords to find matches
    token_words=word_tokenize(text)
    stemmed = [porter.stem(word) for word in token_words]
    clean_text = " ".join(stemmed)    
    found_keywords = [w for w in stemmed_keywords.keys() if any([(syn in clean_text) for syn in stemmed_keywords[w]])]#.split()]
    # this will give a big list of words that may have duplicates
    return found_keywords

def process_keyword_results(entry_keywords_to_transf): # entry_keywords_to_transf[entry.risk_id][word].append(t.name)
    word_groups = {}
    return 0

def preprocess_cv_hazop(entries):
    all_abbrive = []
    for en in entries:
        entry_text = (entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower() + '.'
    return 0

entry_file = 'cv_hazop_all.csv'#'cv_hazop_light_sources.csv'
glossary_file = 'image_processing_terms.txt' #'image_processing_glossary.txt'
keywords = load_keywords(glossary_file)
transformations = parse_transformations('transformations.md')
entries = parse_cv_hazop_entries(entry_file)


syn = list(set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('rain')])))

all_keywords = []
transformations_keywords = {}
for transf in transformations:
    transformations_keywords[transf.name] = []
    t_keywords = find_keywords(transf.name + ' ' + transf.description, keywords)
    transformations_keywords[transf.name] += t_keywords
    all_keywords += t_keywords

# remove keywords not meaningful (appear in at least half of the transformations)
to_remove = []
for k in set(all_keywords):
    if len([t for t in transformations_keywords if k in transformations_keywords[t]]) >= 0.4 * len(transformations):
        to_remove.append(k)

for t in transformations_keywords.keys():
    new_value = [w for w in transformations_keywords[t] if w not in to_remove]
    transformations_keywords[t] = new_value


#text = "too much light"#"Too many shadows"#"Too faint light (in parts of the scene)"
entry_keywords_to_transf = {}
not_found = []
for entry in entries:
    entry_text = (entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower() + '.'
    if 'too' in entry_text or 'very' in entry_text:
    #if entry.risk_id in ['9', '10', '11', '17', '30', '31', '32']:#['1', '2', '3', '4', '5']:
        #print(entry.risk_id)
        #entry_text = (entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower() + '.' # split this and check for only np or ADV ADJ N/ADJ
        #print(entry_text)
        #print(entry.risk_id)
        #check_parcing((entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower())
        #exit()
        # should do it for each
        results = (parce(entry.meaning.lower())).union(parce(entry.consequence.lower()), parce(entry.risk.lower()))
        print(results)
        #if len(results) > 0 and ('too' not in entry_text and 'very' not in entry_text):
        if len(results) == 0:
            not_found.append(entry.risk_id)
        continue
        # lemmetize and find keywords
        entry_keywords = find_keywords(' '.join(results), keywords)
        

        # match with transformations
        entry_keywords_to_transf[entry.risk_id] = {}
        #entry_keywords = entries_keyword[entry.risk_id]
        if len(entry_keywords) == 0:
            continue
        for word in entry_keywords:
            entry_keywords_to_transf[entry.risk_id][word] = []

        # check for additional transformations that cover more keywords
        for t in transformations:
            t_keywords = transformations_keywords[t.name]
            for word in t_keywords:
                if word in entry_keywords:
                    if t not in entry_keywords_to_transf[entry.risk_id][word]:
                        entry_keywords_to_transf[entry.risk_id][word].append(t.name)
        continue
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(entry_text)

        for token in doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop)
        for np in doc.noun_chunks:
            print(np.text)
#patterns = [('increase', NP), ('decrease', NP), ('too', ADJ)]
#increase_words = set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('increase')]))
#decrease_words = set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('decrease')]))
print(entry_keywords_to_transf)
print(not_found)
#print([ss.lemma_names() for ss in wn.synsets('light')])