from parse_cv_hazop import *
from parse_transformations import *
from utils import *
from pattern_matching import *
import random
import pickle
import os
#print(reduce('median filter'))
#print(reduce('filter'))
#print(reduce('blurry image'))
#print(reduce('blur'))
#print(list(set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('spectrum')]))))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# manually checking the similairty approach
entry_keywords_to_transf = {}

glossary_file = 'image_processing_terms.txt' #'image_processing_glossary.txt'
keywords = load_keywords(glossary_file)


'''

model = SentenceTransformer('bert-base-nli-mean-tokens')

entry_1159_match = ['Larger pixel size', 'lower noise level', 'less resolution of observer']
superpixel_match = ['Transform image to their superpixel representation', 'replace pixel within segment by their average color']
compression_match = ['image compression', 'image quality lower bound', 'image quality upper bound']
noise_match = ['Apply camera sensor noise']
trans_matches = [superpixel_match, compression_match, noise_match]

for e_match in entry_1159_match:
    sentence_embeddings1 = model.encode(e_match)
    for t_match in trans_matches:
        for t_m in t_match:
            print(e_match +'; '+ t_m)
            sentence_embeddings2 = model.encode(t_m)
            sim_arr=cosine_similarity(
                [sentence_embeddings1],
                [sentence_embeddings2]
            )
            print(sim_arr)
            print('----------------')
exit()



if os.path.isfile('albumentation.pickle'):
    with open('albumentation.pickle', 'rb') as handle:
        transformations = pickle.load(handle)
else:
    transformations = TransformationList('transformations.csv')
    transformations.match_keywords(keywords)
    with open('albumentation.pickle', 'wb') as handle:
        pickle.dump(transformations, handle, protocol=pickle.HIGHEST_PROTOCOL)

for t in transformations.all_transformations:
    if 'ISO Noise' in t.name:
        print(t.name)
        print(t.match)
        print('------------')
exit()
'''
entry_file = 'exp_entries'#'cv_hazop_all.csv'#'cv_hazop_light_sources.csv'

#if os.path.isfile(entry_file + '.pickle'):
#    with open(entry_file+'.pickle', 'rb') as handle:
#        entries = pickle.load(handle)
#else:
entries = CV_HAZOP_checklist(entry_file+'.csv')
entries.match_keywords(keywords)
with open(entry_file+'.pickle', 'wb') as handle:
    pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)

exit()

location_to_keyword = {}
unions = set()
for location in entries.entries:
    #print(location)
    entries_at_this_loc = [e for e in entries.all_entries if e.location == location]
    all_keywords = set(sum([e.keywords[0] for e in entries_at_this_loc], []))
    if len(unions) > 0:
        unions = unions.intersection(all_keywords)
    else:
        union = all_keywords
    #print(len(all_keywords))
    #print(all_keywords)

    location_to_keyword[location] = all_keywords
    #print('----------------------------------')

#print(len(unions))
#print(unions)
'''
for t in transformations.all_transformations:
    print(t.name)
    print(t.keywords)
    print('-------------------------')


# trying unique keywords
for location1 in location_to_keyword:
    print(location1)
    print('old len: ' + str(len(location_to_keyword[location1])))
    for location2 in location_to_keyword:
        if location1 != location2:
            intersect12 = location_to_keyword[location1].intersection(location_to_keyword[location2])
            location_to_keyword[location1] = location_to_keyword[location1] - intersect12
    print('new len: ' + str(len(location_to_keyword[location1])))
    print(location_to_keyword[location1])
    print('---------------')

exit()
'''
location_to_transf = {}
for location in location_to_keyword:
    print(location)
    print('----------------------------------')
    print('----------------------------------')
    transf = [t.name for t in transformations.all_transformations if len(set(t.keywords).intersection(location_to_keyword[location])) > 0]
    location_to_transf[location] = transf
    all_keywords = location_to_keyword[location]
    all_keywords = [k for k in all_keywords if k != 'image' and k != 'pixel']
    print('Keywords:')
    print(len(all_keywords))
    print(all_keywords)
    print('Transformations:')
    print(len(transf))
    print(transf)
    print('----------------------------------')
    print('---------------per param-------------------')
    # also match parameters
    parameter_keyword = []
    for param in entries.entries[location]:
        print(param)
        
        stemmed=new_reduce(word_tokenize(param.lower()))
        #stemmed = {reduce(word) for word in token_words}
        #print(stemmed)
        param_keywords = []
        for w in keywords.keys():
            for s in keywords[w]:
                #if w == 'resolution':
                #    
                #    print(word_tokenize(s))
                #    print(set(stemmed) >= set(word_tokenize(s)))
                if set(stemmed) >= set(word_tokenize(s)):
                    if len(word_tokenize(s)) > 0:
                        #print(word_tokenize(s), w)
                        param_keywords.append(w)
                        break
        
        entries_with_this_param = [e for e in entries.all_entries if e.location == location and e.parameter == param]
        all_param_keywords = set(sum([e.keywords[0] for e in entries_with_this_param], [])).union(param_keywords)
        all_param_keywords = [k for k in all_param_keywords if k != 'image' and k != 'pixel']
        print('Keywords:')
        print(all_param_keywords)
        #parameter_keyword.append(param)
    
        transf = [t.name for t in transformations.all_transformations if len(set(t.keywords).intersection(location_to_keyword[location])) > 0 and len(set(t.keywords).intersection(all_param_keywords)) > 0]
        location_to_transf[location] = transf
        all_keywords = location_to_keyword[location]
        #print('Keywords:')
        #print(len(all_keywords))
        #print(all_keywords)
        print('Transformations:')
        print(len(transf))
        print(transf)
        print('----------------------------------')

exit()

# match entries with transformations
for entry in entries.all_entries:    
    print('---------------' + entry.risk_id +'-----------------')
    entry_keywords_to_transf[entry.risk_id] = {}
    #entry_keywords = ['camera', 'interference', 'noise']#entry.keywords
    #entry_keywords = entries_keyword[entry.risk_id]
    entry_keywords = list(set(itertools.chain.from_iterable(entry.keywords)))
    if len(entry_keywords) == 0:
        continue
    for word in entry_keywords:
        entry_keywords_to_transf[entry.risk_id][word] = []

    # check for additional transformations that cover more keywords
    for t in transformations.all_transformations:
        #print(t.keywords)
        for word in t.keywords:
            if word == 'object':
                continue
            if word in entry_keywords:
                if t not in entry_keywords_to_transf[entry.risk_id][word]:
                    entry_keywords_to_transf[entry.risk_id][word].append(t.name)
    grouped_results = group_keywords(entry_keywords_to_transf, entry)
    print(grouped_results)

    #nlp = spacy.load("en_core_web_lg")
    #doc = nlp(entry_text)

    #for token in doc:
    #    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #            token.shape_, token.is_alpha, token.is_stop)
    #for np in doc.noun_chunks:
    #    print(np.text)
#patterns = [('increase', NP), ('decrease', NP), ('too', ADJ)]
#increase_words = set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('increase')]))
#decrease_words = set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('decrease')]))


#print(not_found)


#print(entry_keywords_to_transf)
