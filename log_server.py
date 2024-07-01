from flask import Flask, request

app = Flask(__name__)

@app.route('/log', methods=['POST'])
def log():
    data = request.get_json()
    print(data["message"])
    return 'OK'



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5050, debug=True)
else:
    print(__name__)
