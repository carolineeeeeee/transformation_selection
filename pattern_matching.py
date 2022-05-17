import stanza

nlp_d = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

modifiers = ['nmod', 'amod', 'advmod', 'appos', 'compound', 'xcomp']

pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'ADP', 'PROPN']

def in_modifiers(deprel):
    new_rel = deprel.split(':')[0]
    return new_rel in modifiers

def link_modifiers(sent, w):
    previous_words = [x for x in sent.words if x.head == w.id and x.pos in pos_tags] # find a word that has w as head
    result = []

    if len(previous_words) == 0:
        if w.deprel == 'conj':
            paral_word = sent.words[w.head-1]
            paral_result = link_modifiers(sent, paral_word)
            paral_result.remove(paral_word.id)
            paral_result.append(w.id)
            return paral_result
        else:
            return [w.id]
    else:
        for p_w in previous_words:
            if p_w.pos == 'ADP' and p_w.text == 'of':
                result += link_modifiers(sent, p_w)
            elif in_modifiers(p_w.deprel):
                result += link_modifiers(sent, p_w)
        result += [w.id]
        
        if w.deprel == 'conj':
            paral_word = sent.words[w.head-1]
            if 'random' not in paral_word.text:
                paral_result = link_modifiers(sent, paral_word)
                paral_result.remove(paral_word.id)
                return result + paral_result 
        return result
        

def prepare_match(matched, result_type=2):
    matched = [x for x in matched if 'expect' not in x.lemma and '_' not in x.lemma and 'random' not in x.lemma]
    if result_type == 1:
        return ' '.join([x.lemma if x.pos != 'ADJ' and 'amod' not in x.deprel  else x.text for x in matched])
    else:
        return ' '.join([str(x.id) for x in matched])

def parse(text):
    doc = nlp_d(text)
    text_patterns = []
    for sent in doc.sentences:
        #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tpos: {word.pos}' for word in sent.words], sep='\n')
        matched_patterns = []

        paired_words = {}
        for w in sent.words:
            # match modifier
            if w.deprel == 'appos' and w.text == sent.words[w.head-1].text:
                continue
            
            linked_text = link_modifiers(sent, w)
            
            linked_text = [sent.words[i-1] for i in linked_text]
            paired_words[w.id] = linked_text
            
        matched_ids = []
        verbs_advs = [w for w in sent.words if w.pos == 'VERB' or w.pos == 'ADV' or w.pos == 'ADJ']
        if len(verbs_advs) > 0:
            for w in verbs_advs:
                subj = [sub.id for sub in sent.words if sub.head == w.id and 'subj' in sub.deprel and sub.pos != 'PRON']
                obj = [obj.id for obj in sent.words if obj.head == w.id and 'obj' in obj.deprel and obj.pos != 'PRON'] #or 'obl' in 
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

        # match rest for effect
        to_remove = ['observer', 'object', 'other', 'scene', 'image', 'lens']
        for w in sent.words:
            if w.id not in matched_ids:
                if w.id in paired_words and all([x.pos in pos_tags for x in paired_words[w.id]]):
                    if w.pos == 'NOUN' or w.pos == 'PROPN':
                        matched = paired_words[w.id]
                        matched_string = prepare_match(matched, result_type=2)
                        overlap = [x for x in matched_patterns if x in matched_string or matched_string in x]
                        
                        if len(overlap) > 0:
                            overlap.append(matched_string)
                            max_string = max(overlap, key=len) 
                            for x in overlap:
                                if x in matched_patterns and x != max_string:
                                    matched_patterns.remove(x)
                            if len(matched_string) < len(max_string):
                                continue
                            else:
                                matched_string = max_string
                        
                        # check if previous words are propositions
                        if w.id > 2:
                            prev_1 = sent.words[w.id -2]
                            prev_2 = sent.words[w.id -3]
                            if ((prev_1.pos == 'ADP' and prev_1.deprel == 'case') and (prev_1.lemma == 'in' or prev_1.lemma == 'per')) or ((prev_2.pos == 'ADP' and prev_2.deprel == 'case') and (prev_2.lemma == 'in' or prev_2.lemma == 'per')):
                                matched_string = 'in-ADP ' + matched_string

                        matched_patterns.append(matched_string)

            clean_matched_patterns = []
            # remove in/per ADP
            for p in matched_patterns:
                if 'in-ADP' in p:
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
                starting_with_i = [x for x in matched_patterns if any([int(w) == sent.words[i-1].head for w in x.split()])]
                if len(starting_with_i) != 0:
                    ending_with_i = [x for x in matched_patterns if int(max(x.split())) == i-1]
                    if len(ending_with_i) > 0:
                        if starting_with_i[0] == ending_with_i[0]:
                            continue
                        matched_patterns.remove(starting_with_i[0])
                        matched_patterns.remove(ending_with_i[0])
                        matched_patterns.append(ending_with_i[0] + ' ' + str(i) + ' ' + starting_with_i[0])

        # translate things back to text
        for p in matched_patterns:
            text_version = ' '.join([sent.words[int(x)-1].text for x in p.split()])
            text_patterns.append(text_version)
        
    return text_patterns

def parse_transf(transformation):
    results = []
    # first add name and parameter names
    if '_' in transformation.name:
        results.append(' '.join(transformation.name.lower().split('_')))
    else:
        results.append(transformation.name.lower())

    results += parse(transformation.description)
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
            non_algo_related_part = ''
            for t in s.split(':'):
                if not any([w in t for w in algorithm_related]):
                    non_algo_related_part += t + ' '
            if len(non_algo_related_part) == 0:
                entry.matching.append([])
                continue
            else:
                s = non_algo_related_part

        results = parse(s)
        entry.matching.append(results)

    return entry.matching




if __name__ == '__main__':
    parse('Too much light.')