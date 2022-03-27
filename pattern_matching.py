from curses.ascii import NL
from turtle import pos
from nltk import word_tokenize
import spacy
import itertools
from nltk.corpus import wordnet as wn
import re
import stanza

nlp_c = stanza.Pipeline('en', processors='tokenize,pos,constituency') # initialize English neural pipeline
nlp_d = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')


modifiers = ['nmod', 'amod', 'advmod', 'appos', 'acl', 'advcl', 'compound']

pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'ADP']


def link_modifiers(sent, w):
    if w.deprel == 'conj':
        paral_word = sent.words[w.head-1]
        paral_result = link_modifiers(sent, paral_word)
        paral_result.remove(paral_word.id)
        paral_result.append(w.id)
        return paral_result
    else:
        previous_words = [x for x in sent.words if x.head == w.id and x.pos in pos_tags] # find a word that has w as head
        previous_words = [p_w for p_w in previous_words if (p_w.deprel in modifiers) or (p_w.pos == 'ADP' and p_w.text == 'of') or 'subj' in p_w.deprel]
        result = []
        if len(previous_words) == 0:
            return [w.id]
        else:
            for p_w in previous_words:
                if p_w.deprel in modifiers:
                    result += link_modifiers(sent, p_w)
                elif p_w.pos == 'ADP' and p_w.text == 'of':
                    result += link_modifiers(sent, p_w)
                elif 'subj' in p_w.deprel:
                    result += link_modifiers(sent, p_w)
            result += [w.id]
            return result

def parse(text):
    doc = nlp_d(text)
    matched_patterns = []
    for sent in doc.sentences:
        #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tpos: {word.pos}' for word in sent.words], sep='\n')
        paired_words = {}
        for w in sent.words:
            # match modifier
            #if sent.words[w.head-1].pos in pos_tags and w.pos in pos_tags: 
            linked_text = link_modifiers(sent, w)
            linked_text = [sent.words[i-1] for i in linked_text]
            paired_words[w.id] = linked_text
            #print(w.text, [x.text for x in linked_text])
        
        # find VERB/ADV subject and object
        matched_ids = []
        verbs_advs = [w for w in sent.words if w.pos == 'VERB' or w.pos == 'ADV']
        if len(verbs_advs) > 0:
            for w in verbs_advs:
                # find sub and obj
                subj = [sub.id for sub in sent.words if sub.head == w.id and 'subj' in sub.deprel]
                obj = [obj.id for obj in sent.words if obj.head == w.id and 'obj' in obj.deprel]
                # match patterns for action
                if subj != [] and obj != []:
                    for i in subj:
                        for j in obj:
                            matched = paired_words[i] + [w] + paired_words[j]
                            matched_patterns.append(' '.join([x.text for x in matched]))
                            matched_ids += [x.id for x in matched]
                elif subj != [] and obj == []:
                    for i in subj:
                        matched = paired_words[i] + [w] 
                        matched_patterns.append(' '.join([x.text for x in matched]))
                        matched_ids += [x.id for x in matched]
                else:
                    for j in obj:
                        matched =  [w] + paired_words[j]
                        matched_patterns.append(' '.join([x.text for x in matched]))
                        matched_ids += [x.id for x in matched]

        #print(matched_patterns)
        #print(matched_ids)

        # match rest for effect
        for w in sent.words:
            if w.id not in matched_ids:
                if w.id in paired_words and all([x.pos in pos_tags for x in paired_words[w.id]]):
                    if any([x.deprel not in modifiers for x in paired_words[w.id] if x.pos == 'NOUN']): # has a noun that is not a modifier
                        matched = paired_words[w.id]
                        matched_patterns.append(' '.join([x.text for x in matched]))
    #print(matched_patterns)
    return matched_patterns

def parse_transf(transformation):
    results = []
    # first add name and parameter names
    names = []
    names.append(transformation.name)
    for p in transformation.parameters:
        names.append(p.replace('_', ' '))
    results.append(names)
    #print(results)

    # then parse description
    results.append(parse(transformation.description))
    for p in transformation.parameters:
        results.append(parse(transformation.parameters[p]))
    
    #print(results)
    return results

def parse_entry(entry): 
    algorithm_related = ['algorithm', 'detect', 'interpret', 'recognize', 'recognise']
    for s in [entry.meaning, entry.consequence, entry.risk]:
        if s.strip() == '':
            entry.matching.append([])
            continue
        if s.startswith('see') or s.startswith('also: see') or any([w in s for w in algorithm_related]):
            entry.matching.append([])
            continue
        print(s)
        entry.matching.append(parse(s))

    return entry.matching




if __name__ == '__main__':
    parse('Too much light.')