# a collection of cv-hazop entries
# need to deal with abbreviation
# need to deal with see
import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
import re

class CV_HAZOP_entry:
    """A class for transformation information"""
    risk_id = ''
    location = ''
    guide_word = ''
    parameter = ''
    meaning= ''
    consequence = ''
    risk = ''
    
    def __init__(self, info_df):
        self.risk_id = str(info_df['Risk Id'])
        self.location = info_df['Location']
        self.guide_word = info_df['Guide Word']
        self.parameter = info_df['Parameter']
        self.matching = []
        self.keywords = []

        if not pd.isna(info_df['Meaning']):
            self.meaning= info_df['Meaning']
        else:
            self.meaning= ''

        if not pd.isna(info_df['Consequence']):
            self.consequence= info_df['Consequence']
        else:
            self.consequence= ''

        if not pd.isna(info_df['Risk']):
            self.risk= info_df['Risk']
        else:
            self.risk= ''

    def __str__(self):
        results = ''
        results += 'risk_id: ' + self.risk_id + ', \n'
        results += 'location: ' + self.location + ', \n'
        results += 'guide_word: ' + self.guide_word + ', \n'
        results += 'parameter: ' + self.parameter+ ', \n'
        results += 'meaning: ' + self.meaning + ', \n'
        results += 'consequence: ' + self.consequence + ', \n'
        results += 'risk: ' + self.risk 
        return results
    
    def update_abbrv(self, abbrv_dict):
        for k in abbrv_dict:
            self.meaning = (self.meaning.lower()+ ' ').replace(k, abbrv_dict[k])
            self.consequence = (self.consequence.lower() + ' ').replace(k, abbrv_dict[k])
            self.risk = (self.risk.lower() + ' ').replace(k, abbrv_dict[k])
        return 0

class CV_HAZOP_checklist:
    """A collection of all considered CV-HAZOP entries"""

    def __init__(self, filename):
        self.entries = {}
        self.keywords = []
        self.all_entries = []
        self.abbr_replacement = {}
        self.abbr_replacement['l.s.'] = 'light source'
        self.abbr_replacement['temp.'] = 'temporal'
        self.abbr_replacement['transp.'] = 'transparent'
        self.abbr_replacement['pos.'] = 'position'
        self.abbr_replacement['pos '] = 'position '
        self.abbr_replacement['compl.'] = 'complexity'
        self.abbr_replacement['appl.'] = 'application'
        self.abbr_replacement['trans.'] = 'transparent'
        self.abbr_replacement['shad.'] = 'shadow'
        self.abbr_replacement['occl.'] = 'occlusion'
        self.abbr_replacement['calib.'] = 'calibration'
        self.abbr_replacement['alg.'] = 'algorithm'
        self.abbr_replacement['obj.'] = 'object'
        self.abbr_replacement['obs.'] = 'observer'
        self.abbr_replacement['spat.'] = 'spatial'
        self.abbr_replacement['orient.'] = 'orientation'
        self.abbr_replacement['pts.'] = 'points'
        self.abbr_replacement['param.'] = 'parameter'
        self.abbr_replacement['resol.'] = 'resolution'
        self.abbr_replacement['refl.'] = 'reflectance'
        self.abbr_replacement['prop.'] = 'property'
        self.abbr_replacement['pol.'] = 'polarized'
        self.abbr_replacement['fov.'] = 'field of view' 
        self.abbr_replacement['fov'] = 'field of view'
        self.abbr_replacement['num.'] = 'number'  
        self.abbr_replacement['/'] = ' or ' 
        self.abbr_replacement['('] = '' 
        self.abbr_replacement[')'] = '' 
        self.abbr_replacement['"'] = '' 
        self.abbr_replacement['*'] = '' 
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            entry = CV_HAZOP_entry(row)

            # keep only image related entries
            if 'Algorithm' in entry.location:
                continue
            param_to_remove = ['Temporal periodic', 'Temporal aperiodic', 'Before', 'After', 'Faster', 'Slower', 'Early', 'Late']
            if entry.parameter in param_to_remove:
                continue
            if 'Observer' in entry.location and entry.parameter == 'Number':
                continue
            
            if entry.location not in self.entries:
                self.entries[entry.location] = {}
            if entry.parameter not in self.entries[entry.location]:
                self.entries[entry.location][entry.parameter] = {}
            if entry.guide_word not in self.entries[entry.location][entry.parameter]:
                self.entries[entry.location][entry.parameter][entry.guide_word] = []
            entry.update_abbrv(self.abbr_replacement)
            self.entries[entry.location][entry.parameter][entry.guide_word].append(entry)
            self.all_entries.append(entry)

    def keywords_with_see(self):
        list_locations = set([e.location for e in self.all_entries])
        list_parameters = set([e.parameter for e in self.all_entries])
        list_guidewords  = {'No':'No (not none)', 'More':'More (more of, higher)', 'Less':'Less (less of, lower)', 
        'As well as':'As well as', 'Part of':'Part of', 'Reverse':'Reverse', 
        'Other than':'Other than', 'Where else':'Where else', 'Spatial periodic':'Spatial periodic', 
        'Spatial aperiodic':'Spatial aperiodic', 'Close':'Close', 'Remote':'Remote', 'In front of':'In front of', 'Behind':'Behind'}
        list_title = {'meaning':0, 'consequence':1, 'risk':2}
        for e in self.all_entries:
            if e.risk_id in ['1002', '817']:
                continue
            for text, index in [(e.meaning.lower(), 0), (e.consequence.lower(), 1), (e.risk.lower(), 2)]:
                if 'see' in text.split():
                    if len(text.split()) > 5:
                        print(e.risk_id)
                        print(text)
                    title = [list_title[t] for t in list_title if t  in text ]     
                       
                    location = [l for l in list_locations if  l.lower() in  text ]
                    
                    parameter = [p for p in list_parameters if p.lower() in   text ]
                    
                    guideword = [g for g in list_guidewords if g.lower() in  text ]
                    if 'other than expected' in text:
                        guideword.remove('Other than')

                    if len(text.split()) > 5:
                        print(title, location, parameter, guideword)
                    if title == [] and location == [] and parameter == [] and guideword == []:
                        continue

                    if title == []:
                        title.append(index)     
                    if location == []:
                        location.append(e.location)
                    if parameter == []:
                        parameter.append(e.parameter)
                    if guideword == []: 
                        guideword.append(e.guide_word)
                    
                    for l in location:
                        for p in parameter:
                            for g in guideword:
                                for t in title:
                                    #print(l,p,g, t)
                                    try:
                                        corresponding_entry = self.entries[l][p][list_guidewords[g]]
                                        # if a number is specified
                                        number = [int(i) for i in text if i.isdigit()]
                                        if number == []:
                                            for c_e in corresponding_entry:
                                                if len(c_e.keywords) > t:
                                                    #e.keywords[t] += c_e.keywords[t]
                                                    e.keywords.append(c_e.keywords[t])
                                        else:
                                            if len(corresponding_entry[number[0]].keywords) > t:
                                                e.keywords.append(corresponding_entry[number[0]].keywords[t])
                                    except:
                                        continue

    def print_abbrv(self):
        self.abbr = []
        for entry in self.all_entries:
            entry_text = ''
            if entry.meaning.endswith('.'):
                entry_text += entry.meaning[:-1]
            else:
                entry_text += entry.consequence
            entry_text += ' '

            if entry.consequence.endswith('.'):
                entry_text += entry.consequence[:-1]
            else:
                entry_text += entry.consequence
            entry_text += ' '

            if entry.risk.endswith('.'):
                entry_text += entry.risk[:-1]
            else:
                entry_text += entry.risk
            words = entry_text.split()
            for w in words:
                if '.' in w:
                    self.abbr.append(w)
            #exit()
        print(set(self.abbr))
    

if __name__ == '__main__':
    entry_file = 'cv_hazop_all.csv'
    cv_hazop = CV_HAZOP_checklist(entry_file)
    #cv_hazop.print_abbrv()


