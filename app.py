from flask import Flask, jsonify, request
from lex_sem import is_Similar
from feature_test import is_Safe
import nltk
nltk.download('words')

app = Flask(__name__)



@app.route('/get-status', methods=['POST'])
def echo():
    data = request.get_json()
    url=data["url"]
    is_similar=is_Similar(url)
    if(is_similar):
        warning=f"This site closely resembles a known trusted site: {is_similar}"
        return jsonify({"message": True,"warnings":[warning]})
    try:
        is_safe=is_Safe(url)
    except Exception as e:
        print("Exception catched",e)
        return jsonify({"error":True})
    if(is_safe):
        return jsonify({
        "message": True,
        "warnings": is_safe
        })


    return jsonify({"message": False})


