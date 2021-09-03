import os
import xml.etree.ElementTree as ET

sample_list = []

def save_xml_text(user_text):
    # user inputs a string
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    # os.path.join is used so that paths work in every operating system
    save_user_path = os.path.join(fileDir, '..', 'data', 'xml')

    with open(os.path.join(save_user_path, 'sample_xml.txt'), 'w+', encoding='utf-8') as f:
       f.write(user_text)

def etree_reader(path, file):
    """
    :param path: the path into amalgum dataset
    :param file: the file name in the folder: amalgum_genre_docxxx
    :return: an element tree object
    """
    file += '.xml'
    if os.path.exists(os.path.join(path, file)):
        tree = ET.parse(os.path.join(path, file))
        root = tree.getroot()
        for child in root:
            for child_a in child:
                # for child_b in child_a:
                #     print(child_b.text)
                for i in child_a.itertext():
                    sample_list.append(i)

        new_list = ' '.join(sample_list)
        new_string = "".join(str(x) for x in new_list)
        save_xml_text(new_string)
    else:
        print(os.path.join(path, file))
        print('file not found')
        pass

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data')


etree_reader(path=test_path, file="enwiki-20181001-corpus")