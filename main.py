# import numpy.random.common
# import numpy.random.bounded_integers
# import numpy.random.entropy
import nltk
nltk.download('names')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('universal_tagset')
nltk.download('dependency_treebank')
nltk.download('punkt')
nltk.download('pros_cons')

from flask_cors import CORS
from bias_visualisation_app import app

CORS(app)


if __name__ == "__main__":
    # Only for debugging while developing
    # app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
    app.run(debug=True)
