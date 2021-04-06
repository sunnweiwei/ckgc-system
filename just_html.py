from flask import Flask, request, render_template

app = Flask(__name__, template_folder='./static/templates')


@app.route('/chat')
def webchat():
    return render_template('inner_index.html')


@app.route('/webchat')
def webwebchat():
    return render_template('inner_index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
    # server = WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
