import xml.etree.ElementTree as ET
import os


path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data', 'enwiki-20181001-corpus.xml')


context = ET.iterparse(test_path, events=('end', ))

title = 0
for event, elem in context:
    while title < 10:
        if elem.tag == 'article':
            #title = elem.find('name').text
            filename = format(str(title) + ".xml")
            title += 1
            print(title)
            with open(filename, 'wb') as f:
                f.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                f.write(ET.tostring(elem))