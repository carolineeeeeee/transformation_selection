from parse_cv_hazop import *
from parse_transformations import *
from utils import *
from pattern_matching import *
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

#print(reduce('median filter'))
#print(reduce('filter'))
#print(reduce('blurry image'))
#print(reduce('blur'))
#print(list(set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('spectrum')]))))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# manually checking the similairty approach
entry_keywords_to_transf = {}


#1. order of words matter a little bit: illumination change VS change brightness -> 0.51,  change illumination VS change brightness -> 0.47. If i use another model that doesn't care about this, the other model struggles with too much light VS change brightness (one way to do this is take max of two models, it's a bit weird)
#2. what to do with single nouns? those create a lot of false positives, for example, pixel

BERT_model = SentenceTransformer('all-distilroberta-v1') # TODO: this one or the other one?
BERT_model_2 = SentenceTransformer('all-mpnet-base-v2') # TODO: this one or the other one?


'''
print('from model 1:')

sim_arr=cosine_similarity(
            [BERT_model.encode('dark scene')],
            [BERT_model.encode('change brightness')]
        )
print('too faint light VS change brightness: ' + str(sim_arr))

print('from model 2:')
sim_arr=cosine_similarity(
            [BERT_model_2.encode('dark scene')],
            [BERT_model_2.encode('change brightness')]
        )
print('too faint light VS change brightness: ' + str(sim_arr))

sim_arr=cosine_similarity(
            [BERT_model.encode('too much light')],
            [BERT_model.encode('change brightness')]
        )
print('too much light VS change brightness: ' + str(sim_arr))
sim_arr=cosine_similarity(
            [BERT_model.encode('illumination change')],
            [BERT_model.encode('change brightness')]
        )
print('illumination change VS change brightness: ' + str(sim_arr))
sim_arr=cosine_similarity(
            [BERT_model.encode('change illumination')],
            [BERT_model.encode('change brightness')]
        )
print('change illumination VS change brightness: ' + str(sim_arr))

# illumination change will not  match with change brightness but change illumination will
BERT_model_2 = SentenceTransformer('all-mpnet-base-v2') # TODO: this one or the other one?

print('from model 2:')
sim_arr=cosine_similarity(
            [BERT_model_2.encode('too faint light')],
            [BERT_model_2.encode('change brightness')]
        )
print('too faint light VS change brightness: ' + str(sim_arr))
sim_arr=cosine_similarity(
            [BERT_model_2.encode('too much light')],
            [BERT_model_2.encode('change brightness')]
        )
print('too much light VS change brightness: ' + str(sim_arr))
sim_arr=cosine_similarity(
            [BERT_model_2.encode('illumination change')],
            [BERT_model_2.encode('change brightness')]
        )
print('illumination change VS change brightness: ' + str(sim_arr))
sim_arr=cosine_similarity(
            [BERT_model_2.encode('change illumination')],
            [BERT_model_2.encode('change brightness')]
        )
print('change illumination VS change brightness: ' + str(sim_arr))


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

'''
def match(library_name, entry_file):
    if os.path.isfile(library_name + '.pickle'):
        with open(library_name+'.pickle', 'rb') as handle:
            transformations = pickle.load(handle)
    else:
        transformations = TransformationList('transformations.csv', lib_name=library_name)
        transformations.match_keywords()
        with open(library_name+'.pickle', 'wb') as handle:
            pickle.dump(transformations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #if os.path.isfile(entry_file + '.pickle'):
    #    with open(entry_file+'.pickle', 'rb') as handle:
    #        entries = pickle.load(handle)
    #else:
    entries = CV_HAZOP_checklist(entry_file+'.csv')
    entries.match_keywords()
    with open(entry_file+'.pickle', 'wb') as handle:
        pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #return
    #exp_entries_id = ['3', '86', '124', '141', '199', '200', '204', '211', '296', '371', '421', '478', '537', '611', '651', '827', '869', '910', '1017', '1120', '1159']
    #exp_entries_id = ['1159']
    exp_entries = entries.all_entries #random.sample(entries.all_entries, 20)#[e for e in entries.all_entries if e.risk_id in exp_entries_id]
    # most used common words after in ADP(case)
    #prop = list(itertools.chain.from_iterable([e.prop for e in exp_entries]))
    #print(prop)
    #words = (' '.join(prop)).split()
    #word_count = []
    #for w in set(words):
    #    word_count.append((w, words.count(w)))
    #    if words.count(w) >= 5:
    #        print(w)
    #print(word_count)
    #exit()
    # TODO: merge scene, view, image, object
    # synonym of image
    #print(list(set(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets('image')]))))
    # matching
    similarity_threshold = 0.5
    match_results = {}
    for e in exp_entries:
        #if e.risk_id not in ['124', '199', '200', '296', '421']:#, '603', '602']:
        #    continue
        print(e.risk_id)
        print(e.matching)
        #exit()
        #if e.risk_id != '1017':
        #    continue
        match_results[e.risk_id] = {}
        e_text = list(itertools.chain.from_iterable(e.matching))
        print(e_text)
        sentence_embeddings1 = BERT_model.encode(e_text)
        sentence_embeddings1_2 = BERT_model_2.encode(e_text)
        #print(sentence_embeddings1.shape)
        for t in transformations.all_transformations:
            #print(t.name)
            #print(t.match)
            if t.match == []:
                continue
            sentence_embeddings2 = BERT_model.encode(t.match)
            sentence_embeddings2_2 = BERT_model_2.encode(t.match)
            #print(sentence_embeddings2.shape)
            sim_arr_1=cosine_similarity(
                sentence_embeddings1,
                sentence_embeddings2
            )
            sim_arr_2=cosine_similarity(
                sentence_embeddings1_2,
                sentence_embeddings2_2
            )
            sim_arr = np.maximum(sim_arr_1, sim_arr_2)
            #if 'Blur' in t.name:
            #    print(t.name)
            #    print(t.match)
            #print(sim_arr)
            matched = np.argwhere(sim_arr >= similarity_threshold)
            for pair in matched:
                if e_text[pair[0]] not in match_results[e.risk_id]:
                    match_results[e.risk_id][e_text[pair[0]]] = []
                match_results[e.risk_id][e_text[pair[0]]].append((t.name, t.match[pair[1]]))
            #exit()
        print(match_results[e.risk_id])
    with open(library_name + '_eval.pickle', 'wb') as handle:
        pickle.dump((match_results, entries, transformations), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return match_results, entries, transformations

def evaluation(library_name):
    #if os.path.isfile(library_name + '_eval.pickle'):
    #    with open(library_name + '_eval.pickle', 'rb') as handle:
    #        match_results, entries, transformations = pickle.load(handle)
    #else:
        #match_results, entries, transformations = match(library_name, 'cv_hazop_all')
    match_results, entries, transformations = match(library_name, 'exp_entries')
    #match_results, entries, transformations = match(library_name, 'cv_hazop_all')
    
    
    # bar chart for how many are in each
    locations = []
    matched_counts = []
    all_covered_counts = []
    for loc in entries.entries:
        print(loc)
        if '-' in loc:
            loc_fixed = loc.replace("-", "-\n")
            locations.append(loc_fixed)
        else:
            locations.append(loc)
        #matched_count = 0
        entries_at_loc = [x for x in entries.all_entries if x.location == loc]
        #print(len(entries_at_loc))

        matched_count = len([x for x in entries_at_loc if match_results[x.risk_id] != {}])
        print('some covered: ' + str(matched_count) + '/' + str(len(entries_at_loc)))
        matched_counts.append(matched_count/len(entries_at_loc))

        all_covered = len([x for x in entries_at_loc if len(match_results[x.risk_id]) == len(sum(x.matching[:-1], []))])
        print('all covered: ' + str(all_covered) + '/' + str(len(entries_at_loc)))
        all_covered_counts.append(all_covered/len(entries_at_loc))


    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(all_covered_counts))
    br2 = [x + barWidth for x in br1]
    #br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, matched_counts, color ='b', width = barWidth,
            edgecolor ='grey', label ='partically covered')
    plt.bar(br2, all_covered_counts, color ='g', width = barWidth,
            edgecolor ='grey', label ='fully covered')
    #plt.bar(br3, CSE, color ='b', width = barWidth,
    #        edgecolor ='grey', label ='CSE')
    
    # Adding Xticks
    plt.xlabel('Location', fontweight ='bold', fontsize = 20)
    plt.ylabel('m_l', fontweight ='bold', fontsize = 20)
    plt.xticks([r + barWidth for r in range(len(locations))],
            locations, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([0, 1])
    plt.legend(prop={'size': 20})
    plt.show()

    # pie chart for the one most covered, which parameter
    loc_1 = 'Observer - Electronics'
    for loc in entries.entries:
        if loc != loc_1:
            continue
        params = []
        matched_counts = []
        all_covered_counts = []
        nothing_covered_all = []
        for param in entries.entries[loc]:
            print(param)
            params.append(param)
            entries_at_param = [x for x in entries.all_entries if x.location == loc and x.parameter == param]
            #print(len(entries_at_loc))
            matched_count = len([x for x in entries_at_param if match_results[x.risk_id] != {}])
            strp = '\shortstack{'
            #print('some covered: ' + str(matched_count) + '/' + str(len(entries_at_param)))
            strp += 'partially: ' + str(matched_count) + '/' + str(len(entries_at_param)) + '\\'
            matched_counts.append(matched_count/len(entries_at_param))

            all_covered = len([x for x in entries_at_param if len(match_results[x.risk_id]) == len(sum(x.matching[:-1], []))])
            #print('all covered: ' + str(all_covered) + '/' + str(len(entries_at_param)))
            strp += 'fully: ' + str(all_covered) + '/' + str(len(entries_at_param)) + '}'
            all_covered_counts.append(all_covered/len(entries_at_param))
            print(strp)

            nothing_covered = [x for x in entries_at_param if len(match_results[x.risk_id]) == 0]
            #print(nothing_covered)
            nothing_covered_all += nothing_covered
        print(len(nothing_covered_all))
        print(len([x for x in nothing_covered_all if 'No' in x.guide_word]))

        y = np.array(matched_counts)
        plt.pie(y, labels=params, autopct=lambda p: '{:.0f}%'.format(p), shadow=True)
        plt.title('partically covered entries in ' + loc)
        plt.show() 

        y = np.array(all_covered_counts)
        plt.pie(y, labels=params, autopct=lambda p: '{:.0f}%'.format(p ), shadow=True)
        plt.title('fully covered entries in ' + loc)
        plt.show() 

if __name__ == '__main__':
    evaluation('albumentations')
    evaluation('torchvision')
    #['338', '361', nothing
    # '467', '541', '542', '543', objs
    # '213', see as well as (2)
    # '554', see more and less
    #  '602', '603', objects in front of another
    # '1151'] overexposure?
