# a collection of transformations
from utils import *

class Transformation:
	"""A class for transformation information"""
	line = ''
	name = ''
	source = ''
	description = ''

	# TODO: later update format
	def __init__(self, info_line):
		self.keywords = []
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


	def __str__(self):
		results = ''
		results += 'name: ' + self.name + ', \n'
		results += 'source: ' + self.source + ', \n'
		results += 'description: ' + self.description + ', \n'
		results += 'effect: ' + str(self.effect)
		return results

def parse_transformations(filename, keywords):
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
	
	# match keywords
	all_keywords = []
	transformations_keywords = {}
	for transf in transformations:
		transformations_keywords[transf.name] = []
		t_keywords = find_keywords([transf.name, transf.description], keywords, is_transf=True)
		transformations_keywords[transf.name] += t_keywords
		all_keywords += t_keywords
		

	# remove keywords not meaningful (appear in at least half of the transformations)
	to_remove = []
	for k in set(all_keywords):
		if len([t for t in transformations_keywords if k in transformations_keywords[t]]) >= 0.4 * len(transformations):
			to_remove.append(k)

	for t in transformations:
		new_value = [w for w in transformations_keywords[t.name] if w not in to_remove]
		t.keywords = new_value
		transformations_keywords[t.name] = new_value

	#print(transformations_keywords['Random Brightness'])
	#print(transformations_keywords['ISO Noise'])
	#print(transformations_keywords['Random Snow'])
	return transformations
