import os
import numpy as np
import pandas as pd



def save_obj(obj, name):
    path_parent = os.path.dirname(os.getcwd())
    save_df_path = os.path.join(path_parent, 'static', 'user_downloads', name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

save_obj(df, 'test_df')