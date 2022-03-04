from parse_cv_hazop import *
from parse_transformations import *
from utils import *
from pattern_matching import *


entry_file = 'cv_hazop_all.csv'#'cv_hazop_light_sources.csv'
glossary_file = 'image_processing_terms.txt' #'image_processing_glossary.txt'
keywords = load_keywords(glossary_file)
transformations = parse_transformations('transformations.md', keywords)
entries = CV_HAZOP_checklist(entry_file)

#check_parcing('object more transparent than expected')
#check_parcing('some processing is much faster than expected')
#exit()

entry_keywords_to_transf = {}
not_found = []
cur_meaning = ''
for entry in entries.all_entries:
    if entry.risk_id in ['1', '2', '3', '4', '5']:
        #entry_text = (entry.meaning + '. ' + entry.consequence + '. ' + entry.risk).lower() + '.'
        #print('--------------------------------')
        #print(entry_text)
        results = parce(entry)
        


        #print(entry.matching)
        #print('--------------------------------')

        #if entry.risk.strip() != cur_meaning:
        #   results = parce(entry)
        #cur_meaning = entry.risk.strip()
        #
        # lemmetize and find keywords
        entry_keywords = find_keywords(' '.join(results), keywords)
        print(entry_keywords)
        # TODO: deal with see
        continue
        # match with transformations
        entry_keywords_to_transf[entry.risk_id] = {}
        #entry_keywords = entries_keyword[entry.risk_id]
        if len(entry_keywords) == 0:
            continue
        for word in entry_keywords:
            entry_keywords_to_transf[entry.risk_id][word] = []

        # check for additional transformations that cover more keywords
        for t in transformations:
            for word in t.keywords:
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