from flask import jsonify, request

def handle_incoming_request():
    # You don't necessarily need to process the request body here.
    # But if you want to access the data, use request.json
    # data = request.json
    data = request.get_json()
    id = data.get("id")
    email = data.get("email")

    return jsonify({"id": id, "email": email}), 200
    
    # Just return a success message and status code
    # return "Success!", 200