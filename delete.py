import os
import shutil

fileDir = os.path.dirname(os.path.realpath('__file__'))
txt_dir = os.path.join(fileDir, 'bias_visualisation_app')


def delete_all_in_dir(folder_path):

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


delete_all_in_dir(os.path.join(txt_dir, 'data', 'user_uploads'))

delete_all_in_dir(os.path.join(txt_dir, 'static', 'user_uploads'))

delete_all_in_dir(os.path.join(txt_dir, 'static', 'user_downloads'))