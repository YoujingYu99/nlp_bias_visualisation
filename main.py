from flask_cors import CORS
from bias_visualisation_app import app

CORS(app)


if __name__ == "__main__":
    # Only for debugging while developing
    # app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
    app.run(debug=True)
