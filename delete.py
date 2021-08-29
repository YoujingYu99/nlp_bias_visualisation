import os
import shutil

fileDir = os.path.dirname(os.path.realpath('__file__'))
txt_dir = os.path.join(fileDir, 'bias_visualisation_app')


def delete_all_in_dir(folder_path_list):
    for folder_path in folder_path_list:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def delete_all_with_ext(folder_path, ext_list):
    files_in_directory = os.listdir(folder_path)
    for ext in ext_list:
        filtered_files = [file for file in files_in_directory if file.endswith(ext) and os.path.splitext(os.path.basename(file))[0] != 'nothing_here']
        for file in filtered_files:
            path_to_file = os.path.join(folder_path, file)
            os.remove(path_to_file)




delete_all_in_dir(folder_path_list=[os.path.join(txt_dir, 'data', 'user_uploads'), os.path.join(txt_dir, 'static', 'user_uploads'), os.path.join(txt_dir, 'static', 'user_downloads')])

delete_all_in_dir(folder_path_list=[os.path.join(txt_dir, 'utils', 'test')])

delete_all_with_ext(folder_path=os.path.join(txt_dir, 'static'), ext_list=['.html', '.png', '.txt', '.csv'])
