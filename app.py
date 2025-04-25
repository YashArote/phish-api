import tracemalloc
tracemalloc.start()
from flask import Flask, jsonify, request
from lex_sem import is_Similar
from feature_test import is_Safe
import atexit

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

@app.route('/memory-usage')
def memory_usage():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    top_lines = []
    for stat in top_stats[:10]:
        top_lines.append(str(stat))

    return jsonify({"top_memory_lines": top_lines})
if __name__ == '__main__':  
   app.run(host='0.0.0.0')