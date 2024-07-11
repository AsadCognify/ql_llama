import requests
from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def pong():
    response = requests.get(url="http://localhost:5000/ping")
    
    try:
        return response
    
    except Exception as e:
        return make_response(jsonify({"message": "Unable to establish connection"}), 512)



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5050, debug=True)
else:
    print(__name__)
