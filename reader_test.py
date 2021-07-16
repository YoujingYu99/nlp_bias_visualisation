from readers import *

p = 'amalgum/amalgum_balanced/tsv'
f = 'amalgum_academic_doc000'
f_read = tsv_reader(p, f)
for row in f_read:
    print(row)

"""
['#Text=1. Introduction']
['1-1', '0-2', '1.', '_', '_', '_', '_']
['1-2', '3-15', 'Introduction', 'abstract', 'new', 'ana', '3-7']
[]
['#Text=As a kind of on – off valve , gate valves are widely used in various process industries .']
['2-1', '16-18', 'As', '_', '_', '_', '_']
['2-2', '19-20', 'a', '_', '_', '_', '_']
['2-3', '21-25', 'kind', '_', '_', '_', '_']
['2-4', '26-28', 'of', '_', '_', '_', '_']
['2-5', '29-31', 'on', '_', '_', '_', '_']
['2-6', '32-33', '–', '_', '_', '_', '_']
['2-7', '34-37', 'off', '_', '_', '_', '_']
['2-8', '38-43', 'valve', 'object', 'new', 'coref', '3-18[11_0]']
['2-9', '44-45', ',', '_', '_', '_', '_']
['2-10', '46-50', 'gate', 'object|object[4]', 'new|new[4]', 'coref|coref', '3-4[7_4]|4-15']
['2-11', '51-57', 'valves', 'object[4]', 'new[4]', '_', '_']
['2-12', '58-61', 'are', '_', '_', '_', '_']
['2-13', '62-68', 'widely', '_', '_', '_', '_']
['2-14', '69-73', 'used', '_', '_', '_', '_']
['2-15', '74-76', 'in', '_', '_', '_', '_']
['2-16', '77-84', 'various', 'abstract[6]', 'new[6]', '_', '_']
['2-17', '85-92', 'process', 'event|abstract[6]', 'new|new[6]', 'coref', '6-11[33_0]']
['2-18', '93-103', 'industries', 'abstract[6]', 'new[6]', '_', '_']
['2-19', '104-105', '.', '_', '_', '_', '_']
[]
"""


p = 'amalgum/amalgum/dep'
f = 'amalgum_academic_doc000'
tokenlists = conllu_reader(p, f)
for tokenlist in tokenlists:
    print(list(tokenlist))
    break

"""
[{'id': 1, 'form': '1.', 'lemma': '@ord@', 'upos': 'X', 'xpos': 'LS', 'feats': None, 'head': 2, 'deprel': 'dep', 'deps': None, 'misc': None}, 
{'id': 2, 'form': 'Introduction', 'lemma': 'introduction', 'upos': 'NOUN', 'xpos': 'NN', 'feats': {'Number': 'Sing'}, 'head': 0, 'deprel': 'root', 'deps': None, 'misc': None}]
"""


p = 'amalgum/amalgum/xml'
f = 'amalgum_academic_doc000'
tree = etree_reader(p, f)
print(tree.getroot())

"""
<Element 'text' at 0x7f9262c06230>
"""