from os import path
import pickle
import sys
# set recursion limit
sys.setrecursionlimit(10000)

def load_obj(name):
    save_df_path = path.join(path.dirname(__file__), "..\\static\\")
    with open(save_df_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_user_uploads(name):
    upload_df_path = path.join(path.dirname(__file__), "..\\static\\user_uploads\\")
    with open(upload_df_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

file1 = load_obj(name="total_dataframe")
print('success!')
print(type(file1))
file2 = load_obj_user_uploads(name="total_dataframe_user_uploads")

print(type(file2))