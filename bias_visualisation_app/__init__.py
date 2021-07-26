from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

from bias_visualisation_app import routes