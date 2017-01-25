from bottle import route, run
from predict import *
from urllib.parse import unquote

@route('/<query>')
def index(query):   
    return {'result': predict(unquote(query), 10)}

run(host='localhost', port=8080)