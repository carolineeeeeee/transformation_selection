#import albumentations
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
s=set(stopwords.words('english'))
import string
import pandas as pd
from nltk import word_tokenize, pos_tag
import itertools

WORDS_REMOVED = [ 'scene', 'object', 'input', 'output'] # when I finish CV-HAZOP we can do a word count to remove these


class Transformation:
	"""A class for transformation information"""
	line = ''
	name = ''
	source = ''
	description = ''
	

	def __init__(self, info_line):
		self.line = info_line
		self.CV_location_parameter = [] # [(light sources, intensity), (Medium, Transparency)]
		self.CV_guideword = [] # [(Less, No), (All)] loop both of them with indices
		self.effect = []
		infos = info_line.split('|')
		if len(infos) < 7:
			raise Exception('not enough columns, expected 7 got ' + str(len(infos)) + ': ' + info_line)
		self.name = infos[1].strip()
		self.source = infos[2].strip()
		self.description = infos[3].strip()
		list_loc_param = (infos[4].strip()).split(';')
		for item in list_loc_param:
			loc, param = item.split(',')
			loc = loc.strip()
			param = param.strip()
			(self.CV_location_parameter).append((loc, param))
		list_guide_words = (infos[5].strip()).split(';')
		for item in list_guide_words:
			guide_words = tuple(item.strip().split(',')) 
			self.CV_guideword.append(guide_words)
		effects = infos[6].strip().split(',')
		for word in effects:
			if word != '':
				self.effect.append(word.strip())

		self.words = []
		self.get_words()
		#self.two_grams = []
		#self.get_two_grams()


	def __str__(self):
		results = ''
		results += 'name: ' + self.name + ', \n'
		results += 'source: ' + self.source + ', \n'
		results += 'description: ' + self.description + ', \n'
		results += 'CV-HAZOP localtion, parameter: '
		for item in self.CV_location_parameter:
			results += '(' + item[0] + ', ' + item[1] + ')'
		results += ', \n'
		results += 'CV-HAZOP guideword: ' + str(self.CV_guideword) + ', \n'
		results += 'effect: ' + str(self.effect)
		return results

	
	def get_words(self):
		description=self.name+' ' + self.description 
		description = description.translate(str.maketrans('', '', string.punctuation))
		words = [porter.stem(x.lower()) for x in description.split() if porter.stem(x.lower()) not in s and porter.stem(x.lower()) not in WORDS_REMOVED] 
		self.words = words
		

	#def get_two_grams(self):
	#	self.two_grams = ngrams(self.words, 2)
	

	def check_entry(self, location, parameter, guideword):
		for i in range(len(self.CV_location_parameter)):
			if self.CV_location_parameter[i][0] == location and self.CV_location_parameter[i][1] == parameter:
				for word in self.CV_guideword[i]:
					if word in guideword or word == 'All':
						return True
		#for pair in self.CV_location_parameter:
		#	if pair[0] == location and pair[1] == parameter:
		#		for word in self.CV_guideword:
		#			if word in guideword:
		#				return True
		#			elif word == 'All':
		#				return True
		return False

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
		
		#infos = info_line.split('|')
		#if len(infos) < 7:
		#	raise Exception('not enough columns, expected 7 got ' + str(len(infos)) + ': ' + info_line)
		self.risk_id = str(info_df['Risk Id'])
		self.location = info_df['Location']
		self.guide_word = info_df['Guide Word']
		self.parameter = info_df['Parameter']

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

		self.words = []
		self.get_words()
		#self.two_grams = []
		#self.get_two_grams()

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

	
	def get_words(self):
		description = self.consequence + ' ' + self.risk
		description = description.translate(str.maketrans('', '', string.punctuation))
		words = [porter.stem(x.lower()) for x in description.split() if porter.stem(x.lower()) not in s and porter.stem(x.lower()) not in WORDS_REMOVED] 
		self.words = words

	#def get_two_grams(self):
	#	self.two_grams = ngrams(self.words, 2)
	

	def check_entry(self, location, parameter, guideword):
		return location == self.location and parameter == self.parameter and guideword in self.guide_word

def tokenize(text):
	# filter and keep only verb and noun
	description = text.translate(str.maketrans('', '', string.punctuation))
	words = [porter.stem(x.lower()) for x in description.split() if porter.stem(x.lower()) not in s] 

	return words

def verb_and_noun(text):
	tagged_words = pos_tag(word_tokenize(text))
	v_and_n = [w[0] for w in tagged_words if w[1].startswith('N') or w[1].startswith('V')]
	no_punc = [w for w in v_and_n if w not in string.punctuation]
	lemmatize_and_no_stop = [porter.stem(x.lower()) for x in no_punc if porter.stem(x.lower()) not in s] 
	return lemmatize_and_no_stop


def stem(text):
	token_words=word_tokenize(text)
	stemmed = [porter.stem(word) for word in token_words]
	clean_text = " ".join(stemmed)
	return clean_text

'''
def find_keywords(text, stemmed_keywords): # TODO: matching 'noise' VS 'Gaussian noise'?
	token_words=word_tokenize(text)
	results = []
	for w in token_words:
		#print(w)
		synonyms = list(itertools.chain.from_iterable([ss.lemma_names() for ss in wn.synsets(w)]))
		synonyms = [porter.stem(s) for s in synonyms] + [w] # including the original word
		matched = [s for s in stemmed_keywords if s in synonyms]
		if matched != []:
			results += matched
	return list(set(results))

	#stemmed = [porter.stem(word) for word in token_words]
	#clean_text = " ".join(stemmed)
	#return [w for w in stemmed_keywords if w in clean_text.split()]

'''
def find_keywords(text, stemmed_keywords): 
	token_words=word_tokenize(text)
	stemmed = [porter.stem(word) for word in token_words]
	clean_text = " ".join(stemmed)
	return [w for w in stemmed_keywords if w in clean_text.split()]


def parse_transformations(filename):
	transformations = []
	f = open(filename, "r")
	line = f.readline()
	while line:
		if '|' in line:
			break
		line = f.readline()
	line = f.readline()
	line = f.readline()# column name + format line

	while line:
		if 'END' in line:
			break
		trans_entry = Transformation(line)
		transformations.append(trans_entry)
		line = f.readline()
	return transformations

def parse_cv_hazop_entries(filename):
	entries = []
	df = pd.read_csv(filename)
	for index, row in df.iterrows():
		entry = CV_HAZOP_entry(row)
		#print(entry)
		entries.append(entry)
	return entries
	#f = open(filename, "r")
	#line = f.readline()
	#while line:
	#	if 'Risk Id' in line:
	#		break
	#	line = f.readline()
	#line = f.readline()
	#line = f.readline()# column name + format line
	#while line:
	#	one_entry = CV_HAZOP_entry(line)
	#	entries.append(one_entry)
	#	line = f.readline()
	#return entries

def entry_matching(transformations, entries):
	#print('----------------------Matching entries info---------------------------')
	# match transformations using location, parameter, guide word
	entry_to_transf = {}
	for entry in entries:
		entry_to_transf[entry.risk_id] = [t for t in transformations if t.check_entry(entry.location, entry.parameter, entry.guide_word)]
	#for e in entry_to_transf.keys():
	#	print('Entry number ' + e + ' has possible transformations:')
	#	print([t.name for t in entry_to_transf[e]])
	#print('----------------------Matching effect info---------------------------') # change this name to something else, we can say effect is something not mentioned in description
	
	# match specific effect
	for entry in entries:
		for t in transformations:
			# match transformations with effect
			for key in t.effect:
				if key in entry.words:
					if t not in entry_to_transf[entry.risk_id]:
						entry_to_transf[entry.risk_id].append(t)
						#print('Entry number ' + entry.risk_id + " has one more possible transformation: " + t.name)
	return entry_to_transf

def keyword_matching(entry_to_transf, entries_keyword, transformations_keyword, transformations):
	entry_keywords_to_transf = {}
	for entry_id in entry_to_transf.keys():
		entry_keywords_to_transf[entry_id] = {}
		entry_keywords = entries_keyword[entry_id]
		if len(entry_keywords) == 0:
			continue
		for word in entry_keywords:
			entry_keywords_to_transf[entry_id][word] = []
		
		# check whether the keywords are already covered by the previous step
		for t in entry_to_transf[entry_id]:
			t_keywords = transformations_keyword[t.name]
			for word in t_keywords:
				if word in entries_keyword[entry_id]:
					entry_keywords_to_transf[entry_id][word].append(t)

		# check for additional transformations that cover more keywords
		for t in transformations:
			t_keywords = transformations_keyword[t.name]
			for word in t_keywords:
				if word in entries_keyword[entry_id]:
					if t not in entry_keywords_to_transf[entry_id][word]:
						entry_keywords_to_transf[entry_id][word].append(t)

	return entry_keywords_to_transf
	'''	
	#print('----------------------Matching description info---------------------------')
	for entry in entries:
		# find a list of keywords
		entry_keywords = keyword_appearance[entry.risk_id]
		#entry_keywords = find_keywords(entry.consequence + ', ' + entry.risk, stemmed_keywords)
		if len(entry_keywords) == 0:
			continue
		#print("entry " + str(entry.risk_id) + " has keywords: " + str(entry_keywords))
		found_keyword = []
		for t in transformations:
			if 'Noise' in t.name:
				continue
			transf_keywords = find_keywords(t.name+', ' +t.description, stemmed_keywords)
			intersect = [x for x in transf_keywords if x in entry_keywords]
			if len(intersect) > 0:
				found_keyword += intersect
				if t not in entry_to_transf[entry.risk_id]:
					entry_to_transf[entry.risk_id].append(t)
					print('found transformation '+ t.name+ ' for keyword ' + str(intersect))
			# match transformations with description
			#words_intersection = set([x for x in t.words if x in entry.words])
			#two_gram_intersection = set([x for x in t.two_grams if x in entry.two_grams])
			#if len(words_intersection) > 1 or len(two_gram_intersection) > 0:
				# takes the maximum match, if some ties, randomly pick one? Or we keep all the ones
				#if t not in entry_to_transf[entry.risk_id]:
					#entry_to_transf[entry.risk_id].append(t)
					#print('Entry number ' + entry.risk_id + " has one more possible transformation: " + t.name)
					#print(words_intersection, two_gram_intersection)
		not_found_keywords = [x for x in entry_keywords if x not in found_keyword]
		if len(not_found_keywords) > 0 :
			print('List of unfound keywords: ' + str(not_found_keywords))
	'''


def find_combination(transformations, entries):
	# what to do with combination of transformations?
	# it should only be considered if all the entries are in the list
	# remove from options
	valid_transformations = []
	for t in transformations:
		if len(t.CV_location_parameter) > 1:
			entries_matched = [i for i in range(len(t.CV_location_parameter)) if len([e for e in entries if e.location == t.CV_location_parameter[i][0] and e.parameter == t.CV_location_parameter[i][1]]) > 0]
			if len(entries_matched) == len(t.CV_location_parameter):
				valid_transformations.append(t)
		else:
			valid_transformations.append(t)
	return valid_transformations

def find_cv_hazop_keywords(cv_hazop_entries, keywords_file, transformations):
	# load image processing keywords
	f = open(keywords_file, "r")
	#keywords = f.read().split('\n')
	#keywords = [w.strip() for w in keywords if '#' not in w and w != '']
	keywords = {}

	for wordline in f.readlines():
		if '#' in wordline:
			continue
		if len(wordline.strip()) == 0:
			continue
		word = wordline.split(':')[0]
		clean_words = stem(word)
		if clean_words not in keywords.keys():
			keywords[clean_words] = wordline
	
	entries_keywords = {}
	for entry in cv_hazop_entries:
		entries_keywords[entry.risk_id] = []
		entries_keywords[entry.risk_id] += find_keywords(entry.consequence + ' ' + entry.risk, keywords.keys())
		#entries_keywords[entry.risk_id] += find_keywords(entry.risk, keywords.keys())
	#print(entries_keywords)

	transformations_keywords = {}
	for transf in transformations:
		transformations_keywords[transf.name] = []
		transformations_keywords[transf.name] += find_keywords(transf.name + ' ' + transf.description, keywords.keys())
		#transformations_keywords[transf.name] += find_keywords(transf.description, keywords.keys())

	return keywords, entries_keywords, transformations_keywords
	# find distribution of these keywords 
	#df = pd.read_csv(cv_hazop_file)
	#word_count = {}
	#grouped_loc = df.groupby(df.Location)
	#location_values = df[["Location"]].values.ravel()
	#unique_locations =  pd.unique(location_values)
	# removing words that appear in the location and parameter
	#all_parameter_values = df[["Parameter"]].values.ravel()
	#unique_parameters = pd.unique(all_parameter_values)
	#stemmed_loc_param = [stem(w) for w in unique_locations] + [stem(w) for w in unique_parameters]
	#to_remove = []
	#for w in stemmed_loc_param:
	#	to_remove += find_keywords(w, keywords.keys())
	#print(to_remove)
	#for w in set(to_remove):
	#	keywords.pop(w, None)
	#print(keywords)
	#exit()

	
	'''
	
	for loc in unique_locations:
		if 'Algorithm' in loc:
			continue
		#word_count[loc] = {}
		location = grouped_loc.get_group(loc)
		#separating the entries by parameters
		grouped = location.groupby(df.Parameter)
		column_values = location[["Parameter"]].values.ravel()
		unique_values =  pd.unique(column_values)
		for param in unique_values:
			# do word count
			#word_count[loc][param] = {}
			group = grouped.get_group(param)
			for index, row in group.iterrows():
				# remove not image relevant ones
				param_to_remove = ['Temporal periodic', 'Temporal aperiodic', 'Before', 'After', 'Faster', 'Slower', 'Early', 'Late']
				if row['Parameter'] in param_to_remove:
					continue
				if 'Observer' in loc and row['Parameter'] == 'Number':
					continue
				words = []
				if type(row['Consequence']) is str:
					words += find_keywords(row['Consequence'], keywords.keys())
					#words += verb_and_noun(row['Consequence'])
				if type(row['Risk']) is str:
					words += find_keywords(row['Risk'], keywords.keys())
					#words += verb_and_noun(row['Risk'])
				for w in words:
					#if w not in words_to_remove:
					if w not in keyword_appearance.keys():
						keyword_appearance[w] = {}
					if loc not in keyword_appearance[w].keys():
						keyword_appearance[w][loc] = {}
					if param not in keyword_appearance[w][loc].keys():
						keyword_appearance[w][loc][param] = 1
					else:
						keyword_appearance[w][loc][param] += 1
					
					#if w not in word_count[loc][param]:
					#	word_count[loc][param][w] = 1
					#else:
					#	word_count[loc][param][w] += 1
	#print(word_count)
	#print(keyword_appearance)
	# analyze distribution of keywords
	print(keyword_appearance.keys())
	#for w in keyword_appearance.keys():
	#	print(w)
	#	print(keyword_appearance[w])
	# remove common keywords
	'''

def parse_final_results(entry_to_transf, keyword_results, entries, keywords):
	for e in entries:
		print('------------------Risk ' + e.risk_id+'---------------------')
		#print(found_keywords)
		#print([t.name for t in entry_to_transf[e.risk_id]])
		matched_transformations = [t.name for t in entry_to_transf[e.risk_id]]
		if len(matched_transformations) == 0:
			print('No transformations found.')
		else:
			print('Found transformations for the entry: ' + str(matched_transformations))		

		print('\n+++ keywords matched\n')
		# adding transformations matched using key words:
		found_keywords = list(keyword_results[e.risk_id].keys())
		not_matched_keywords = []
		keyword_transformations = []
		for k in found_keywords:
			cur_keyword = [t.name for t in keyword_results[e.risk_id][k]] 
			if len(cur_keyword) != 0:
				keyword_transformations += cur_keyword
				print('For ' + keywords[k].strip() + ': ' + str(cur_keyword))
			else:
				not_matched_keywords.append(k) 

		#if len(matched) > 0:
		#	matched_transformations += [t.name for t in matched]
		
		if len(not_matched_keywords) > 0:
			print('\n+++ Found potential missing transformations:\n')
			for k in not_matched_keywords:
				print(keywords[k]) # add description
	return 0

if __name__ == '__main__':
	entry_file = 'exp_entries.csv'
	glossary_file = 'image_processing_terms.txt' #'image_processing_glossary.txt'

	transformations = parse_transformations('transformations.md')
	entries = parse_cv_hazop_entries(entry_file)

	non_combination_transformations = find_combination(transformations, entries) # remove combinations

	entry_to_transf = entry_matching(non_combination_transformations, entries)	# match location

	# match keywords
	keywords, entries_keyword, transformations_keyword = find_cv_hazop_keywords(entries, glossary_file, non_combination_transformations)
	#print(entries_keyword)
	#print(transformations_keyword)
	#exit()
	entry_keywords_to_transf = keyword_matching(entry_to_transf, entries_keyword, transformations_keyword, non_combination_transformations)
	parse_final_results(entry_to_transf, entry_keywords_to_transf, entries, keywords)