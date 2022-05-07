#from curses.ascii import NL
#from turtle import pos
#from nltk import word_tokenize
#import spacy
#import itertools
from nltk.corpus import wordnet as wn
import re
import stanza

#nlp_c = stanza.Pipeline('en', processors='tokenize,pos,constituency') # initialize English neural pipeline
nlp_d = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

modifiers = ['nmod', 'amod', 'advmod', 'appos', 'compound', 'xcomp']

pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'ADP', 'PROPN']

def in_modifiers(deprel):
    new_rel = deprel.split(':')[0]
    return new_rel in modifiers

def link_modifiers(sent, w):
    #print('in link_modifiers')
    #print(w.text)
    previous_words = [x for x in sent.words if x.head == w.id and x.pos in pos_tags] # find a word that has w as head
    #print([(w.text, w.deprel) for w in previous_words])
    #previous_words = [p_w for p_w in previous_words if in_modifiers(p_w.deprel) or (p_w.pos == 'ADP' and p_w.text == 'of') or 'subj' in p_w.deprel]
    #print([w.text for w in previous_words])
    result = []

    if len(previous_words) == 0:
        if w.deprel == 'conj':
            #print(w.text)
            paral_word = sent.words[w.head-1]
            #print('paral word: ' + paral_word.text)
            paral_result = link_modifiers(sent, paral_word)
            #print('paral word result: ' + ' '.join([sent.words[x-1].text for x in paral_result]))
            paral_result.remove(paral_word.id)
            paral_result.append(w.id)
            #print(' '.join([sent.words[x-1].text for x in paral_result]))
            return paral_result
        else:
            return [w.id]
    else:
        for p_w in previous_words:
            #print(p_w.text)
            #if 'random' in p_w.text:
            #    continue
            if p_w.pos == 'ADP' and p_w.text == 'of':
                #print('of')
                result += link_modifiers(sent, p_w)
                #print(result)
            elif in_modifiers(p_w.deprel):
                #print('found modifier')
                #if p_w.deprel == 'nmod' and sent.words[p_w.id-2].deprel == 'case' and sent.words[p_w.id-2].lemma != 'of': # separate case except for of that makes a word
                #    continue
                result += link_modifiers(sent, p_w)
                #print(result)
            #elif 'subj' in p_w.deprel:
            #    print('subj')
            #    result += link_modifiers(sent, p_w)
            #    print(result)
            #elif 'root' in p_w.deprel and p_w.pos == 'ADJ':
            #    print('root')
            #    result += link_modifiers(sent, p_w)
        result += [w.id]
        
        if w.deprel == 'conj':
            #print(w.text)
            paral_word = sent.words[w.head-1]
            if 'random' not in paral_word.text:
                #print('paral word: ' + paral_word.text)
                paral_result = link_modifiers(sent, paral_word)
                #print('paral word result: ' + ' '.join([sent.words[x-1].text for x in paral_result]))
                paral_result.remove(paral_word.id)
                #paral_result.append(w.id)
                #print(' '.join([sent.words[x-1].text for x in paral_result]))
                #return paral_result
                return result + paral_result 
        return result
        

def prepare_match(matched, result_type=2):
    matched = [x for x in matched if 'expect' not in x.lemma and '_' not in x.lemma and 'random' not in x.lemma]
    if result_type == 1:
        return ' '.join([x.lemma if x.pos != 'ADJ' and 'amod' not in x.deprel  else x.text for x in matched])
    else:
        return ' '.join([str(x.id) for x in matched])

def parse(text):
    #text='objects is behind another'
    doc = nlp_d(text)
    #doc_2 = nlp_c(text)
    #for sentence in doc_2.sentences:
    #    print(sentence.constituency)
    matched_patterns = []
    prop_patterns = []
    for sent in doc.sentences:
        #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tpos: {word.pos}' for word in sent.words], sep='\n')
        
        paired_words = {}
        for w in sent.words:
            # match modifier
            if w.deprel == 'appos' and w.text == sent.words[w.head-1].text:
                continue
            #if sent.words[w.head-1].pos in pos_tags and w.pos in pos_tags: 
            #print(w.text, w.id)
            #previous_words = [x for x in sent.words if x.head == w.id]# and x.pos in pos_tags] # find a word that has w as head
            #print([w.text for w in previous_words])
            linked_text = link_modifiers(sent, w)
            #linked_text.sort()
            
            linked_text = [sent.words[i-1] for i in linked_text]
            paired_words[w.id] = linked_text
            #paired_words[w.id] = [w.text for w in linked_text]
            #print(w.text, [x.text for x in linked_text])
        #print(paired_words)
        #continue
        # deal with conj
        #for w in sent.words:
        #    conj = []
        #    if w.deprel == 'conj':
        #       conj.append(w)
            
        #continue
        # find VERB/ADV subject and object
        matched_ids = []
        verbs_advs = [w for w in sent.words if w.pos == 'VERB' or w.pos == 'ADV' or w.pos == 'ADJ']
        #verbs_advs = [w for w in verbs_advs if not in_modifiers(w.deprel)]
        if len(verbs_advs) > 0:
            for w in verbs_advs:
                #print('------------------')
                #print(w.text)
                # find sub and obj
                subj = [sub.id for sub in sent.words if sub.head == w.id and 'subj' in sub.deprel and sub.pos != 'PRON']
                #print('subj:')
                #print([sub.text for sub in sent.words if sub.head == w.id and 'subj' in sub.deprel and sub.pos != 'PRON'])
                obj = [obj.id for obj in sent.words if obj.head == w.id and 'obj' in obj.deprel and obj.pos != 'PRON'] #or 'obl' in obj.deprel) and obj.pos != 'PRON']
                #print('obj:')
                #1`     print([obj.text for obj in sent.words if obj.head == w.id and 'obj' in obj.deprel and obj.pos != 'PRON'])
                # match patterns for action
                
                if subj != [] and obj != []:
                    for i in subj:
                        conj_subj = [conj.id for conj in sent.words if conj.head == i and 'conj' in conj.deprel]
                        conj_subj.append(i)
                        for k in conj_subj:
                            for j in obj:
                                matched = paired_words[k] + paired_words[w.id] + paired_words[j]
                                matched_patterns.append(prepare_match(matched))
                                matched_ids += [x.id for x in matched]
                                conj_obj = [conj.id for conj in sent.words if conj.head == j and 'conj' in conj.deprel]
                                for conj_w in conj_obj:
                                    matched =  paired_words[k] + paired_words[w.id] + paired_words[conj_w]
                                    matched_patterns.append(prepare_match(matched))
                                    matched_ids += [x.id for x in matched]
                elif subj != [] and obj == []:
                    for i in subj:
                        conj_subj = [conj.id for conj in sent.words if conj.head == i and 'conj' in conj.deprel]
                        conj_subj.append(i)
                        for k in conj_subj:
                            matched = paired_words[k] + paired_words[w.id] 
                            matched_patterns.append(prepare_match(matched))
                            matched_ids += [x.id for x in matched]
                else:
                    for j in obj:
                        matched =  paired_words[w.id] + paired_words[j]
                        matched_patterns.append(prepare_match(matched))
                        matched_ids += [x.id for x in matched]
                        conj_obj = [conj.id for conj in sent.words if conj.head == j and 'conj' in conj.deprel]
                        for conj_w in conj_obj:
                            matched =  paired_words[w.id] + paired_words[conj_w]
                            matched_patterns.append(prepare_match(matched))
                            matched_ids += [x.id for x in matched]

        #print(matched_patterns)
        #print(matched_ids)

        # match rest for effect
        to_remove = ['observer', 'object', 'other', 'scene', 'image', 'lens']
        for w in sent.words:
            if w.id not in matched_ids:
                if w.id in paired_words and all([x.pos in pos_tags for x in paired_words[w.id]]):
                    #if any([not in_modifiers(x.deprel) for x in paired_words[w.id] if x.pos == 'NOUN']): # has a noun that is not a modifier
                    if w.pos == 'NOUN' or w.pos == 'PROPN':
                        #print(w.lemma)
                        matched = paired_words[w.id]
                        #if len(matched) <= 1 and 'exposure' not in w.text: # TODO: how do we deal with single word? it creates a lot of false positives
                        #    continue
                        #if len(matched) == 1 and matched in to_remove:
                        #    continue
                        
                        #matched_list = 
                        matched_string = prepare_match(matched, result_type=2)
                        overlap = [x for x in matched_patterns if x in matched_string or matched_string in x]
                        
                        
                        if len(overlap) > 0:
                            overlap.append(matched_string)
                            max_string = max(overlap, key=len) 
                            for x in overlap:
                                if x in matched_patterns and x != max_string:
                                    #print('removing ' + x)
                                    matched_patterns.remove(x)
                            #continue
                            if len(matched_string) < len(max_string):
                                #print('asdf')
                                continue
                            else:
                                matched_string = max_string
                        
                        # check if previous words are propositions
                        if w.id > 2:
                            prev_1 = sent.words[w.id -2]
                            prev_2 = sent.words[w.id -3]
                            if ((prev_1.pos == 'ADP' and prev_1.deprel == 'case') and (prev_1.lemma == 'in' or prev_1.lemma == 'per')) or ((prev_2.pos == 'ADP' and prev_2.deprel == 'case') and (prev_2.lemma == 'in' or prev_2.lemma == 'per')):
                                #print(w.lemma)
                                matched_string = 'in-ADP ' + matched_string
                        #        to_remove = ['observer', 'object', 'other', 'scene', 'image', 'lens']
                        #    
                        #        if any([x in to_remove for x in matched_string.split()]):
                        #            continue
                            #if any([x in w.text for x in ['image', 'sensor', 'pixel', 'scene']]):
                            #prop_patterns.append(matched_string)
                        #else:
                        
                        matched_patterns.append(matched_string)
                        #print(matched_patterns)
    #matched_patterns = [p for p in matched_patterns if 'in-ADP' not in p or any([w in p for w in ['observer', 'object', 'other', 'scene', 'image', 'lens']])]
    #print(matched_patterns)
    #print(matched_patterns)
        clean_matched_patterns = []
        # remove in/per ADP
        for p in matched_patterns:
            if 'in-ADP' in p:
                print('here')
                if any([sent.words[int(x)-1].lemma in to_remove for x in p.split()[1:]]):
                    continue
                else:
                    clean_matched_patterns.append(p[7:])
            else:
                clean_matched_patterns.append(p)
        matched_patterns = clean_matched_patterns

        # check for relationship between NPs matched
        for i in range(len(sent.words)):
            if sent.words[i-1].pos == 'ADP' and sent.words[i-1].deprel == 'case': # find a proposition
                # look for phrases right after (linked by dependency)
                #print("adp " + str(i))
                starting_with_i = [x for x in matched_patterns if any([int(w) == sent.words[i-1].head for w in x.split()])]#[x for x in matched_patterns if int(min(x.split())) == i+1] 
                if len(starting_with_i) != 0:
                    #print('phrase after prop')
                    #print(starting_with_i)
                    
                    # case: just one prop
                    ending_with_i = [x for x in matched_patterns if int(max(x.split())) == i-1]
                    if len(ending_with_i) > 0:
                        #print('just a prop')
                        #print(starting_with_i[0])
                        #print(ending_with_i[0])
                        if starting_with_i[0] == ending_with_i[0]:
                            continue
                        matched_patterns.remove(starting_with_i[0])
                        matched_patterns.remove(ending_with_i[0])
                        matched_patterns.append(ending_with_i[0] + ' ' + str(i) + ' ' + starting_with_i[0])
                        #print(ending_with_i[0] + ' ' + str(i) + ' ' + starting_with_i[0])
                    # case: adj/adv prop
                    #ending_with_i = [x for x in matched_patterns if int(max(x.split())) == i-1]

        # translate things back to text
        text_patterns = []
        for p in matched_patterns:
            text_version = ' '.join([sent.words[int(x)-1].text for x in p.split()])
            text_patterns.append(text_version)

        '''
        for p in matched_patterns:
            list_of_words = [int(x) for x in p.split()]
            list_of_words.sort()
            if list_of_words[-1] >= len(sent.words)-1:
                continue
            last_word = sent.words[list_of_words[-1]-1]
            next_word_after_last = sent.words[list_of_words[-1]]
            if next_word_after_last.pos == 'ADP' and last_word.pos == 'ADJ'.deprel == 'case':
                # check if anything is connected
                following_this = []
        '''
            
        
    return text_patterns

def parse_transf(transformation):
    results = []
    # first add name and parameter names
    #names = [] 
    if '_' in transformation.name:
        results.append(' '.join(transformation.name.lower().split('_')))
    else:
        results.append(transformation.name.lower())
    
    
    #for p in transformation.parameters:
    #    names.append(p.replace('_', ' '))
    #results.append(names)
    #print(results)

    # then parse description
    results += parse(transformation.description)
    #for p in transformation.parameters:
    #    results+=parse(transformation.parameters[p])
    
    #print(results)
    return results

def parse_entry(entry): 
    algorithm_related = ['algorithm', 'detect', 'interpret', 'recognize', 'recognition', 'false positive', 'false negative']
    for s in [entry.meaning, entry.consequence, entry.risk]:
        if s.strip() == '':
            entry.matching.append([])
            continue
        if s.startswith('see') or s.startswith('also: see'):
            entry.matching.append([])
            continue
        
        if any([w in s for w in algorithm_related]):
            #print('in algo related')
            non_algo_related_part = ''
            #print( s.split(':'))
            for t in s.split(':'):
                if not any([w in t for w in algorithm_related]):
                    non_algo_related_part += t + ' '
            #print(non_algo_related_part)
            if len(non_algo_related_part) == 0:
                entry.matching.append([])
                continue
            else:
                s = non_algo_related_part
        #print(s)
        #exit()
        results = parse(s)
        entry.matching.append(results)
    #exit()
    # add info about the combination
    #comb_text = re.sub(" [\(\[].*?[\)\]]", "", entry.guide_word) +' ' + entry.parameter + ' ' + entry.location
    #entry.matching.append([comb_text.lower()])
    #print(entry.prop)
    return entry.matching




if __name__ == '__main__':
    parse('Too much light.')