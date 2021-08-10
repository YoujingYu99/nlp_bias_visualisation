from flask import Flask
from flask_cors import CORS
import os

path_parent = os.path.dirname(os.getcwd())
save_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_downloads')
print(save_path)

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = save_path

from bias_visualisation_app import routes