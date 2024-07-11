import requests
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def pong():
    try:
        response = requests.get(url="http://localhost:5000/ping", timeout=1)
        return make_response(jsonify({"message": "Ping sucesful."}), 200)
    except Exception as e:
        return make_response(jsonify({"message": "Unable to establish connection"}), 512)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)
else:
    print(__name__)

