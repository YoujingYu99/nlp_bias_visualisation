
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.tree import ParentedTree

# (previously using nltk=3.4.5)
# now using 3.3.0

# nltk.download()

dep_parser = CoreNLPDependencyParser()
pos_tagger = CoreNLPParser(tagtype='pos')

def triplet_extraction(input_sent, output=['parse_tree', 'spo', 'result']):
    # parse the input sentence with Stanford CoreNLP Parser
    pos_type = pos_tagger.tag(input_sent.split())
    parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
    dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))
    # Extract subject, predicate and object
    subject = extract_subject(parse_tree)
    predicate = extract_predicate(parse_tree)
    objects = extract_object(parse_tree)
    if 'parse_tree' in output:
        print('---Parse Tree---')
        parse_tree.pretty_print()
    if 'spo' in output:
        print('---Subject---')
        print(subject)
        print('---Predicate---')
        print(predicate)
        print('---Object---')
        print(objects)
    if 'result' in output:
        print('---Result---')
        print(' '.join([subject[0], predicate[0], objects[0]]))

def extract_subject(parse_tree):
    # Extract the first noun found in NP_subtree
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = [t[0], extract_attr(t)]
            # Avoid empty or repeated values
            if output != [] and output not in subject:
                subject.append(output)
    if len(subject) != 0: return subject[0]
    else: return ['']

def extract_predicate(parse_tree):
    # Extract the deepest(last) verb foybd ub VP_subtree
    output, predicate = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label().startswith('VB')):
            output = [t[0], extract_attr(t)]
            if output != [] and output not in predicate:
                predicate.append(output)
    if len(predicate) != 0: return predicate[-1]
    else: return ['']

def extract_object(parse_tree):
    # Extract the first noun or first adjective in NP, PP, ADP siblings of VP_subtree
    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = [word[0], extract_attr(word)]
            if output != [] and output not in objects:
                objects.append(output)
    if len(objects) != 0: return objects[0]
    else: return ['']

def extract_attr(word):
    attrs = []
    # Search among the word's siblings
    if word.label().startswith('JJ'):
        for p in word.parent():
            if p.label() == 'RB':
                attrs.append(p[0])
    elif word.label().startswith('NN'):
        for p in word.parent():
            if p.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                attrs.append(p[0])
    elif word.label().startswith('VB'):
        for p in word.parent():
            if p.label() == 'ADVP':
                attrs.append(p[0])
    # Search among the word's uncles
    if word.label().startswith('NN') or word.label().startswith('JJ'):
        for p in word.parent().parent():
            if p.label() == 'PP' and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    elif word.label().startswith('VB'):
        for p in word.parent().parent():
            if p.label().startswith('VB') and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    return attrs


print('Ready!')
triplet_extraction('A rare black squirrel has become a regular visitor to a suburban garden')