from parse_cv_hazop import *
from parse_transformations import *
from utils import *
from pattern_matching import *
import pickle
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

entry_keywords_to_transf = {}

BERT_model = SentenceTransformer('all-distilroberta-v1') 
BERT_model_2 = SentenceTransformer('all-mpnet-base-v2') 

def match(library_name, entry_file):
    if os.path.isfile('outputs/' + library_name + '.pickle'):
        with open('outputs/' + library_name+'.pickle', 'rb') as handle:
            transformations = pickle.load(handle)
    else:
        transformations = TransformationList('inputs/transformations.csv', lib_name=library_name)
        transformations.parse_effect_action()

    with open('outputs/' + library_name+'.pickle', 'wb') as handle:
        pickle.dump('outputs/' + transformations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile('outputs/' + entry_file + '.pickle'):
        with open('outputs/' + entry_file+'.pickle', 'rb') as handle:
            entries = pickle.load(handle)
    else:
        entries = CV_HAZOP_checklist('inputs/' + entry_file+'.csv')
        entries.parse_effect_action()
        with open('outputs/' + entry_file+'.pickle', 'wb') as handle:
            pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    similarity_threshold = 0.5
    match_results = {}
    for e in entries.all_entries:
        print(e.risk_id)
        print(e.matching)
        match_results[e.risk_id] = {}
        e_text = list(itertools.chain.from_iterable(e.matching))
        print(e_text)
        sentence_embeddings1 = BERT_model.encode(e_text)
        sentence_embeddings1_2 = BERT_model_2.encode(e_text)
        for t in transformations.all_transformations:
            if t.match == []:
                continue
            sentence_embeddings2 = BERT_model.encode(t.match)
            sentence_embeddings2_2 = BERT_model_2.encode(t.match)
            sim_arr_1=cosine_similarity(
                sentence_embeddings1,
                sentence_embeddings2
            )
            sim_arr_2=cosine_similarity(
                sentence_embeddings1_2,
                sentence_embeddings2_2
            )
            sim_arr = np.maximum(sim_arr_1, sim_arr_2)
            matched = np.argwhere(sim_arr >= similarity_threshold)
            for pair in matched:
                if e_text[pair[0]] not in match_results[e.risk_id]:
                    match_results[e.risk_id][e_text[pair[0]]] = []
                match_results[e.risk_id][e_text[pair[0]]].append((t.name, t.match[pair[1]]))
            #exit()
        print(match_results[e.risk_id])
    with open('outputs/' + library_name + '_eval.pickle', 'wb') as handle:
        pickle.dump((match_results, entries, transformations), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return match_results, entries, transformations

def evaluation(library_name, cv_hazop_entries='cv_hazop_all'):
    if os.path.isfile('outputs/' + library_name + '_eval.pickle'):
        with open('outputs/' + library_name + '_eval.pickle', 'rb') as handle:
            match_results, entries, transformations = pickle.load(handle)
            #exit()
    else:
        match_results, entries, transformations = match(library_name, cv_hazop_entries)

    locations = []
    for loc in entries.entries:
        print(loc)
        if '-' in loc:
            loc_fixed = loc.replace("-", "-\n")
            locations.append(loc_fixed)
        else:
            locations.append(loc)
        entries_at_loc = [x for x in entries.all_entries if x.location == loc]
        total_scene_changes = 0
        scene_changes_matched = 0
        for e in entries_at_loc:
            scene_changes_e = list(itertools.chain.from_iterable(e.matching))
            total_scene_changes += len(scene_changes_e)
            matched_changes_e = [x for x in scene_changes_e if x in match_results[e.risk_id]]
            scene_changes_matched += len(matched_changes_e)

        all_covered = len([x for x in entries_at_loc if len(match_results[x.risk_id]) == len(sum(x.matching, []))])
        percentage = all_covered/len(entries_at_loc) * 100 
        print('entries covered: '  + str(percentage) + '\%' + ' (' + str(all_covered) + '/' + str(len(entries_at_loc))+ ')')

        percentage = scene_changes_matched/total_scene_changes * 100 
        print('scene changes covered: ' + str(percentage) + '\%' + ' (' + str(scene_changes_matched) + '/' + str(total_scene_changes)+ ')')
       
        
    loc_1 = 'Observer - Electronics'
    for loc in entries.entries:
        if loc != loc_1:
            continue
        print('----------------------')
        for param in entries.entries[loc]:
            print(param)
            entries_at_param = [x for x in entries.all_entries if x.location == loc and x.parameter == param]

            scene_changes_matched = 0
            total_scene_changes = 0
            for e in entries_at_param:
                scene_changes_e = list(itertools.chain.from_iterable(e.matching))
                total_scene_changes += len(scene_changes_e)
                matched_changes_e = [x for x in scene_changes_e if x in match_results[e.risk_id]]
                scene_changes_matched += len(matched_changes_e)

            all_covered = len([x for x in entries_at_param if len(match_results[x.risk_id]) == len(sum(x.matching, []))])
            percentage = all_covered/len(entries_at_param) * 100 
            print('entries covered: '  + str(percentage) + '\%' + ' (' + str(all_covered) + '/' + str(len(entries_at_param))+ ')')

            percentage = scene_changes_matched/total_scene_changes * 100 
            print('scene changes covered: ' + str(percentage) + '\%' + ' (' + str(scene_changes_matched) + '/' + str(total_scene_changes)+ ')')
        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="The automated matching method. It first matches transformations to CV-HAZOP entries then evulate coverage.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cv_hazop_entries", type=str, help="Name of the csv file containing the CV-HAZOP entries (without .csv extension)")
    parser.add_argument("-t", "--library_name", type=str, help="Name of the transformation library to use. Note that it should exist in transformations.csv")
    args = parser.parse_args()
    config = vars(args)
    evaluation( config['library_name'], config['cv_hazop_entries'])
    #print(config)

    #evaluation('albumentations')
    #evaluation('torchvision')

