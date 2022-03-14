from curses.ascii import NL
from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re

# if np we treat as NP, otherwise check adj

nlp = spacy.load("en_core_web_lg")

def parse_text(text):
    #text = 'too bright light or window'
    doc = nlp(text)
    new_text = ''
    for token in doc:
        if token.pos_ == 'PUNCT':
            text += token.text + ' '
        if token.pos_ == 'SPACE':
            continue
        #if len(re.findall(r'\[[^\[\]]*'+token.text + '[^\[\]]*\]', text)) == 0:
        new_text += token.text+ ' (' +  token.pos_ +') ' #re.sub('(' + token.text + ')', '[\g<1> (' + token.pos_ +')]', text)
    
    # calibration: field of view is miss labelled, should be an NP
    if 'field of view' in text:
        words = 'field of view'.split(' ')
        re_pattern = ''
        for word in words:
            re_pattern += '\s*' + word+'\s*\([A-Z]+\)'
        new_text = re.sub('(' + re_pattern + ')', '[(NP) \g<1>]', new_text)

    for np in doc.noun_chunks:
        # calibration: field of view is miss labelled, should be an NP
        if 'field of view' in text:
            if np.text == 'view' or np.text == 'field':
                continue
        words = np.text.split(' ')
        re_pattern = ''
        for word in words:
            re_pattern += '\s*' + word+'\s*\([A-Z]+\)'
        new_text = re.sub('(' + re_pattern + ')', '[(NP) \g<1>]', new_text)

    # Calibration: light source is misclassified
    new_text = new_text.replace('light (ADJ) source (NOUN)', 'light (NOUN) source (NOUN)')
               
    # Calibration: NP of NP is an NP itself
    new_text = re.sub('\[\(NP\)([^\[\]]+)\]\s([^\[\]\(\)]+)\s\(ADP\)\s*\[\(NP\)([^\[\]]+)\]', '[(NP) \g<1> \g<2> (ADP) \g<3>]', new_text)
    new_text = re.sub('([^\[\]\s]+\s\(NOUN\))\s([^\[\]\(\)]+)\s\(ADP\)\s*\[\(NP\)([^\[\]]+)\]', '[(NP) \g<1> \g<2> (ADP) \g<3>]', new_text)

    return new_text.strip()

    # Calibration: include or inside NP?



# increases, decreases, polarizes etc
def vps(parsed_text): # TODO: separate: VERB NOUN+
    #NP VERB NP
    match = re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', '', parsed_text)
    match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', '', parsed_text)
    match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]]+\s*\(VERB\))', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]]+\s*\(VERB\))', '', parsed_text)
    match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\))', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\))', '', parsed_text)
    
    #match += re.findall(r'(\[\(NP\)[^\[\]]+\])?\s*(([^\[\]]+\s*\(ADV\))?\s*[^\[\]]+\s*\(VERB\).*)(\s*\[\(NP\)[^\[\]]+\])?', parsed_text)

    return list(set(match)), parsed_text

# NP1 becomes (ADV) ADJ and causes NP2,..., NPn, NP2 is an effect of ADJ NP1. e.g., Light becomes (too/very) faint and causes camera sensor noise.
def nps(parsed_text):
    results = []
    #nps += re.findall(r'(\[\(NP\)\s*[^\[\]]+\s*\])\s*[^\[\]\(\)]+\s*\(AUX\)\s*([^\[\]\(\)]+\s*\(ADJ\))|(\[\(NP\)\s*[^\[\]]+\s*\])\s*[^\[\]\(\)]+\s*\(AUX\)\s*([^\[\]\(\)]+\s*\(ADV\)\s*[^\[\]\(\)]+\s*\(ADJ\))', parsed_text)

    # Calibration: captured camera noise, increased noise, captured should be ADJ (if verb is inside an NP, it should be an adjective)
    #nps = re.findall(r'(\[\(NP\)\s*[^\[\]]+ \(VERB\)\s*[^\[\]]+ \(NOUN\)[^\[\]]*\])', parsed_text)

    # more contrast (NP that has adj in them) Calibration: some times ADJ is misclassified as ADV
    #nps = re.findall(r'\[\(NP\)(\s*[^\[\]]+ \(ADV\)\s*[^\[\]]+ \(ADJ\))(\s*[^\[\]]+ \(NOUN\)+)\]|\[\(NP\)(\s*[^\[\]]+ \(ADJ\))(\s*[^\[\]]+ \(NOUN\)+)\]|\[\(NP\)(\s*[^\[\]]+ \(ADV\))(\s*[^\[\]]+ \(NOUN\)+)\]', parsed_text)
    #nps += re.findall(r'(\[\(NP\)\s*[^\[\]]+ \(ADV\)\s*[^\[\]]+ \(ADJ\)\s*[^\[\]]+ \(NOUN\)[^\[\]]*\])|(\[\(NP\)\s*[^\[\]]+ \(ADJ\)\s*[^\[\]]+ \(NOUN\)[^\[\]]*\])|(\[\(NP\)\s*[^\[\]]+ \(ADV\)\s*[^\[\]]+ \(NOUN\)[^\[\]]*\])', parsed_text)

    # loss of contrast
    #nps += re.findall(r'(\[\(NP\)[^\[\]]+\]\sof\s\(ADP\)\s*\[\(NP\)[^\[\]]+\])', parsed_text)

    # distortion, blur
    #nps += re.findall(r'^(\[\(NP\)[^\[\]]+\])$|^(\[\(NP\)[^\[\]]+\])\s*[,\.:]|[,\.:](\[\(NP\)[^\[\]]+\])\s*[,\.:]', parsed_text)

    

    #contrast is more
    #nps += re.findall(r'(\[\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]]+ \(ADV\)\s*[^\[\]]+ \(ADJ\))|\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]]+ \(ADJ\))|\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]]+ \(ADV\))|\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]]+ \(VERB\))', parsed_text)

    nps = re.findall(r'(\[\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

    nps += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

    nps += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

    # AUX or ADP, combination of less?
    nps += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

    nps += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\))', parsed_text)
    parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\))', '', parsed_text)

    nps += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(VERB\))', parsed_text)
    parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(VERB\))', '', parsed_text)

    nps += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', '', parsed_text)

    nps += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', parsed_text)
    parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', '', parsed_text)

    nps += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\))', parsed_text)
    parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\))', '', parsed_text)

    nps += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', parsed_text)
    parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', '', parsed_text)

    nps += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', parsed_text)
    parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', '', parsed_text)

    #e.g., ..
    #nps += re.findall(r'e.g.\s*\(ADV\)(.*)', parsed_text)

    nps += re.findall(r'(\[\(NP\)\s*[^\[\]]+\])', parsed_text)
    parsed_text = re.sub('(\[\(NP\)\s*[^\[\]]+\])', '', parsed_text)

    #for i in range(len(nps)):
    #    if isinstance(nps[i], str):
    #        results.append(nps[i])
    #    else:
    #        results.append(max(nps[i], key=len))

    return list(set(nps)), parsed_text
    #print(nps)
    #return [np for np in nps if 'ADJ' in np and 'NOUN' in np]
    for i in range(len(nps)):
        ADJ = None
        N = None
        for m in nps[i]:
            if not ADJ and 'ADJ' in m:
                ADJ = m
            if not N and 'NOUN' in m:
                N = m
        if ADJ and N:
            results.append(ADJ.strip() + ' ' + N.strip())
    
    return results
    '''

    for np in nps: # want ADV ADJ NOUN
        adj = ''
        noun = ''
        ADJs = re.findall(r'([^\[\]\(\)\s]+\s\(ADJ\))|([^\[\]\(\)\s]+\s\(ADV\))', np)
        #print(ADJs)
        
        if len(ADJs) >0:
            for a in ADJs:
                if isinstance(a, tuple):
                    adj += ' '.join(x for x in a if x != '') + ' '
                else:
                    adj += a + ' '
        #print(adj)
        NOUNs = re.findall(r'([^\[\]\(\)\s]+\s*\(NOUN\))', np)
        #if 'light' in np and 'source' in np: # Calibration: light in here is mislabeled as ADJ
        #    noun += 'light source'
        #else:
        if len(NOUNs) >0:
            for a in NOUNs:
                if isinstance(a, tuple):
                    noun += ' '.join(x for x in a if 'light' not in x and x != '')
                else:
                    noun += a + ' '
        #print(noun)
        if adj != '' and noun != '':
            results.append(adj + noun)
    return results
    '''

# something is caused by something
def np_caused_by_np(parsed_text):
    match = re.findall(r'(\[\(NP\)[^\[\]]+\])\s*caused\s*\([A-Z]+\)\sby\s*\([A-Z]+\)\s*(\[\(NP\)[^\[\]]+\])', parsed_text)
    return match

def np_is(parsed_text):
    # if without than expected, AUX need to be present
    match = re.findall(r'\[(\(NP\)[^\[\]]+)\]\s[a-z]+\s\(AUX\)+([^\.\,]*)', parsed_text)
    #nps += re.findall(r'(\[\(NP\)\s*[^\[\]]+\s*\])\s*[^\[\]\(\)]+\s*\(AUX\)\s*([^\[\]\(\)]+\s*\(ADJ\))|(\[\(NP\)\s*[^\[\]]+\s*\])\s*[^\[\]\(\)]+\s*\(AUX\)\s*([^\[\]\(\)]+\s*\(ADV\)\s*[^\[\]\(\)]+\s*\(ADJ\))', parsed_text)

    return match

def than_expected(parsed_text):
    # with than, with or without expected or AUX
    # (old with out groups) match = re.findall(r'\[(\(NP\)[^\[\]]+)\](\s[a-z]+\s\(AUX\))?(.*)(than\s\([A-Z]+\)(\sexpected\s\([A-Z]+\))?)', parsed_text)
    #match = re.findall(r'(\[\(NP\)[^\[\]]+\])(\s[a-z]+\s\(AUX\))?.*((\s[^\[\]]+ \(ADV\))?\s([^\[\]]+ \(ADJ\)))(.*)(than\s\([A-Z]+\)(\sexpected\s\([A-Z]+\))?)', parsed_text)
    match = re.findall(r'(.*) than\s\([A-Z]+\)', parsed_text)
    return match

def eg(parsed_text):
    # with than, with or without expected or AUX
    # (old with out groups) match = re.findall(r'\[(\(NP\)[^\[\]]+)\](\s[a-z]+\s\(AUX\))?(.*)(than\s\([A-Z]+\)(\sexpected\s\([A-Z]+\))?)', parsed_text)
    #match = re.findall(r'(\[\(NP\)[^\[\]]+\])(\s[a-z]+\s\(AUX\))?.*((\s[^\[\]]+ \(ADV\))?\s([^\[\]]+ \(ADJ\)))(.*)(than\s\([A-Z]+\)(\sexpected\s\([A-Z]+\))?)', parsed_text)
    match = re.findall(r'e.g.\s*\(ADV\)(.*)', parsed_text)
    match += re.findall(r'-(.*)', parsed_text)
    return match

    # only NP
    match = re.findall(r'(\[(\(NP\)[^\[\]]+)\]) than\s\([A-Z]+\)', parsed_text)
    if match != []:
        return []

    # something AUX/VERB ADJ than expected, 
    match = re.findall(r'(.*)\s[^\[\]\(\)]+\(AUX\)(.*ADJ.*)than|(.*)\s[^\[\]\(\)]+\(VERB\)(.*ADJ.*)than',parsed_text)

    # something ADJ than expected, e.g., exposure time longer than expected
    if match == []:
        match = re.findall(r'(.*)\s*([^\[\]\(\)]+\s*\(ADJ\).*)than\s\([A-Z]+\)', parsed_text)
    print(match)

    results = []
    for i in range(len(match)):
        NP1 = None
        NP2 = None
        ADJ = None
        for m in match[i]:
            if NP1 is None and 'NP' in m:
                NP1 = m.strip()
            if NP1 and 'NP' in m and m.strip() != NP1:
                NP2 = m.strip()
            if 'ADJ' in m:
                if (ADJ is None) or (ADJ and len(m.strip()) > len(ADJ)):
                    ADJ = m.strip()
        if NP1 and NP2 and ADJ:
            results.append(ADJ + ' ' + NP2 + ' of ' + NP1)
        if NP2 is None and NP1 and ADJ:
            results.append(ADJ + ' ' + NP1 )
    return results

    return match

    results = []
    if len(match) > 0:
        for m in match:
            if 'AUX' in m or 'than' in m or 'expect' in m or m == '':
                continue
            print(m)
            if 'NP' in m:
                m = re.sub('(\([A-Z]+\))', '', m)
                print(m)
        exit()
    return results



def parse(entry): # meaning needs pair of N, ADJ (ADV ADJ)
    # TODO: output pairs of info and match transformations according to the pairs (pairs are each match)
    results = []
    # np (Done)
    entry_text = (entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower() + '.'
    #print(entry_text)
    #if 'more' in entry.guide_word or 'less' in entry.guide_word:
        # TODO: in the end I am only looking for adj and verb + NP
        # adj N, N is adj
        # V N
        # only N
        #print('------------------------------------------')
        #print(entry_text)
    for s in [entry.meaning, entry.consequence, entry.risk]:
        
        if s.strip() == '':
            continue
        if s.startswith('see') or s.startswith('also: see') or 'algorithm' in s:
            entry.matching.append([])
            continue

        parsed_text = parse_text(s.lower())    
        print(parsed_text)
        # once match remove?


        # ADJ: NP? *ADV? (or ADV)* ADJ or*)* NP? need maximum match
        # VERB: NP? *ADV? (or ADV)* VERB or*)* NP? need maximum match
        # NP (leftover)

        match = re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*\[\(NP\)[^\[\]]+\])', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]]+\s*\(VERB\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]]+\s*\(VERB\))', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*[^\[\]]+\s*\(ADV\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\)\s*[^\[\]]+\s*\(ADV\))', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(VERB\))', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADV\)\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('(\[\(NP\)[^\[\]]+\]\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\))', parsed_text)
        parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(ADV\))', '', parsed_text)

        match += re.findall(r'\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(VERB\))', parsed_text)
        parsed_text = re.sub('\[(\(NP\)[^\[\]]+\]\s[a-z]+\s\(AUX\)+\s*[^\[\]\s]+\s*\(VERB\))', '', parsed_text)

        match += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\))', parsed_text)
        parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\))', '', parsed_text)

        match += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', parsed_text)
        parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADJ\))', '', parsed_text)

        match += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', parsed_text)
        parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', '', parsed_text)

        match += re.findall(r'([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', parsed_text)
        parsed_text = re.sub('([^\[\]\(\)\s]+\s*\(CCONJ\)\s*[^\[\]\(\)\s]+\s*\(ADV\)\s*[^\[\]\(\)\s]+\s*\(VERB\))', '', parsed_text)

        #e.g., ..
        #nps += re.findall(r'e.g.\s*\(ADV\)(.*)', parsed_text)

        match += re.findall(r'(\[\(NP\)\s*[^\[\]]+\])', parsed_text)
        parsed_text = re.sub('(\[\(NP\)\s*[^\[\]]+\])', '', parsed_text)
        #results += eg(text)
        #results += than_expected(text)
        #results += np_caused_by_np(text)
        #results += np_is(text)
        results = []
        for text in match:
            text = re.sub('\([^\(\)]+\)', '', text)
            text = re.sub('\[', '', text)
            text = re.sub('\]', '', text)
            text = re.sub('\s+', ' ', text)
            text = text.strip()
            if text not in results:# and 'confuse' not in text: # temp: confuse usually describes the effect of the transformation on algorithms rather than images
                results.append(text)
        entry.matching.append(results)
            
            #if results == []:
            #    print('------------------------------------------')
            #    print(entry.consequence) 
            #    print(text)
            #    print(results)
        #results = from_pattern_to_match(results)
        #print(results)
        #print('------------------------------------------')
    return entry.matching
    if 'more' in entry.guide_word or 'less' in entry.guide_word:
        print('------------------------------------------')
        print(text)
        results += parce_more_less(text)
        print('------------------------------------------')
        #print(set(results))

    # different than expected
    #results += parce_different(doc)

    return set(results)

def from_pattern_to_match(patterns):
    for pattern in patterns:
        # Calibration 1: sometimes light is mislabelled as ADJ
        # Calibration 2: mixing

        print(pattern)

    return 0


# patterns for keyword 'more' and 'less'
def parce_more_less(text):  #TODO: deal with or and /, TODO: NOUN to NP
    # first one is ADV, ADJ, NOUN
    #match = re.findall(r'[^\[\]\s]*\s*\(ADV\)\s*[^\[\]\s]*\s*\(ADJ\)\s*[^\[\]\s]*\s*\(NOUN\)', text)
    # second one ADV, ADJ, NP
    # third one ADV, ADJ or ADJ
    #print(match)
    return re.findall(r'[^\[\]\s]*\s*\(ADV\)\s*[^\[\]\s]*\s*\(ADJ\)\s*[^\[\]\s]*\s*\(NOUN\)', text)
    exit()
    results = []

    for i in range(len(doc)):
        # too ADJ NOUN, e.g., too faint light
        
        if doc[i].pos_ == 'ADV' and i < len(doc)-2:# and (doc[i].text == 'too' or doc[i].text == 'very'):  
            if doc[i+1].pos_=='ADJ' and (doc[i+2].pos_ == 'NOUN' or (doc[i+2].pos_=='ADJ' and doc[i+2].lemma_ == 'light')) : # a calibration because of the mistake of the parser                        
                results.append(doc[i].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)
                print('first: ' + doc[i].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)

        # NOUN ... is too ADJ, e.g., light is too faint
        if doc[i].pos_ == 'AUX' and i < len(doc)-2:
            if doc[i+1].pos_ == 'ADV':# and (doc[i+1].text == 'too' or doc[i+1].text == 'very'): 
                if doc[i+2].pos_ == 'ADJ':
                    # find the noun right before them
                    for j in range(i+1):
                        if doc[i-j].pos_ == 'NOUN':# and doc[i-j].text != 'scene':
                            results.append(doc[i-j].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)    
                            print('second: ' + doc[i-j].text + ' ' + doc[i+1].text + ' ' + doc[i+2].text)
    # NP is more ADJ or ADJ than expected (match normally then look for er, r and more, less)
    more_less = re.findall(r'\(?(.*) than .*\)?', text)
    if len(more_less) > 0:
        results.append(more_less[0])    
        print(more_less)
    
    

    # increases, decreases

    # something caused by, something caused

    
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

##### effects #####
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


