from flask import Flask, request
from qlm import handle_incoming_request
app = Flask(__name__)

@app.route("/finetune", methods=["POST"])
def handle_finetune_request():
  return handle_incoming_request()


if __name__ == "__main__":
  app.run(debug=True)
