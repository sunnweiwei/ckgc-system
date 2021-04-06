from flask import Flask, request, render_template
import json
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
from geventwebsocket.websocket import WebSocket
import hashlib
import requests as req
import os

app = Flask(__name__, template_folder='./static/templates')


@app.route('/webchat')
def webchat():
    return render_template('inner_index.html')


if __name__ == '__main__':
    server = WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
