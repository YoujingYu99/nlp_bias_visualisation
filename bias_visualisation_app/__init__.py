from flask import Flask
import os

path_parent = os.path.dirname(os.getcwd())
save_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_downloads')
debias_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static')

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = save_path
app.config['DEBIAS_FOLDER'] = debias_path
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = "234798238423"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


from bias_visualisation_app import routes

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response