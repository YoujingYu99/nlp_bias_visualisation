import os
import xml.etree.ElementTree as ET


def save_xml_text(file_name, user_text):
    # user inputs a string
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    # os.path.join is used so that paths work in every operating system
    save_user_path = os.path.join(fileDir, '..', 'data', 'enwiki_txt')
    with open(os.path.join(save_user_path, str(file_name) + '.txt'), 'w+', encoding='utf-8') as f:
       f.write(user_text)

def etree_reader(path):
    """
    :param path: the path into amalgum dataset
    :param file: the file name in the folder: amalgum_genre_docxxx
    :return: an element tree object
    """
    xml_files = os.listdir(path)
    file_n = len(xml_files)
    print('{} files being processed'.format(file_n))
    count = 0
    for file in xml_files:
        if file.endswith(".xml"):
            sample_list = []
            tree = ET.parse(os.path.join(path, file))
            root = tree.getroot()
            for child in root:
                for child_a in child:
                    for i in child_a.itertext():
                        sample_list.append(i)
            count += 1

            new_list = ' '.join(sample_list)
            new_string = "".join(str(x) for x in new_list)
            print(new_string)
            save_xml_text(file_name=count, user_text=new_string)

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data', 'enwiki_xml')


etree_reader(path=test_path)