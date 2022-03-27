import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_lg")
doc = nlp(u'More light increases shadow complexity.')
displacy.serve(doc, style='dep')
