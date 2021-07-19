from os import environ
from bias_statements.PcaBiasCalculator import PcaBiasCalculator
from bias_statements.PrecalculatedBiasCalculator import PrecalculatedBiasCalculator
from bias_statements import app
from flask import request, jsonify
import werkzeug
from bias_statements.parse_sentence import parse_sentence

if environ.get("USE_PRECALCULATED_BIASES", "").upper() == "TRUE":
    print("using precalculated biases")
    calculator = PrecalculatedBiasCalculator()
else:
    calculator = PcaBiasCalculator()

neutral_words = [
    "is",
    "was",
    "who",
    "what",
    "where",
    "the",
    "it",
]


@app.route("/")
def index():
    return "OK"


@app.route("/detect", methods=["GET"])
def detect():
    sentence = request.args.get("sentence")
    if not sentence:
        raise werkzeug.exceptions.BadRequest("You must provide a sentence param")
    if len(sentence) > 500:
        raise werkzeug.exceptions.BadRequest(
            "Sentence must be at most 500 characters long"
        )
    objs = parse_sentence(sentence)
    results = []
    for obj in objs:
        token_result = {
            "token": obj["text"],
            "bias": calculator.detect_bias(obj["text"]),
            "parts": [
                {
                    "whitespace": token.whitespace_,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "ent": token.ent_type_,
                    "skip": token.pos_
                    in ["AUX", "ADP", "PUNCT", "SPACE", "DET", "PART", "CCONJ"]
                    or len(token) < 2
                    or token.text.lower() in neutral_words,
                }
                for token in obj["tokens"]
            ],
        }
        results.append(token_result)
    return jsonify({"results": results})

'''The server is written in Python using flask. It exposes a REST API with a single GET endpoint, /detect which requires passing a sentence query param. For example: /detect?sentence=She is a nurse. This endpoint will tokenize the sentence and return biases for each token, like below:'''