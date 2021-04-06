from flask import Flask, request, render_template
import json
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
from geventwebsocket.websocket import WebSocket
import hashlib
import requests as req
import pysolr
import numpy as np
from f1_score import _f1_score
from collections import defaultdict
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import sent_tokenize
from copy import deepcopy
import nltk
import pickle
import random
import string
import time
import os

from nltk.corpus import stopwords
from stopwords import sw

sw = sw.split('\n')
stopwords = stopwords.words('english')
STOPWORDS = set([x for x in STOPWORDS] + stopwords + sw)


def time_stamp():
    return time.strftime("%H:%M:%S", time.localtime())


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


app = Flask(__name__, template_folder='./static/templates')

solr = pysolr.Solr('http://10.102.32.111:8983/solr/wiki_passage/', always_commit=True, timeout=600)
solr.ping()
print('server start')
user_dict = {}
leaderboard = defaultdict(list)
as_wizard = defaultdict(list)
as_apprentice = defaultdict(list)
last_role = {}
starboard = defaultdict(list)
waiting_list = {}
chatting_list = {}
translate_cache = {}
data_saved = defaultdict(list)
struct_data = dict()
line_data = defaultdict(list)
dialog_length = defaultdict(int)
conv_id_list = ['1']
topic_set = list(read_pkl('unseen_topic.pkl'))
no_passage_used = f'<p><b>no_passage_used</b></p><p><input type="radio" id="no_passage_used" name="category" value="no_passage_used" /><label for="no_passage_used">no_passage_used</label></p>'
print('topic length =', len(topic_set))
data_name = 'sp_data_day4'


def translate_api(query):
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    appid = '20201024000597658'
    key = 'CwwEUKXWjfDlbxZcCUl_'
    payload = {'q': query,
               'from': 'auto',
               'to': 'en',
               'appid': appid,
               'salt': '12345'}
    sign = payload['appid'] + payload['q'] + payload['salt'] + key
    hl = hashlib.md5()
    hl.update(sign.encode(encoding='utf-8'))
    sign = hl.hexdigest()
    payload['sign'] = sign
    response = req.get(url, params=payload)
    response = response.json()
    try:
        v = [x['dst'] for x in response['trans_result']]
    except:
        v = []
    return v


def translate(query):
    query = query.split('\n')
    query = [line for line in query if line != '']
    translated_query = ['' for line in query]
    new_translate = []
    new_id = []
    for j, line in enumerate(query):
        if line in translate_cache:
            translated_query[j] = translate_cache[line]
        else:
            new_translate.append(line)
            new_id.append(j)
    if len(new_translate) > 0:
        results = translate_api('\n'.join(new_translate))
        if len(results) == len(new_id):
            for j, res in zip(new_id, results):
                translated_query[j] = res
    for line, translated_line in zip(query, translated_query):
        translate_cache[line] = translated_line
    return '\n'.join(translated_query)


def write_json():
    time_name = time_stamp().split(':')[0]
    if not os.path.exists(f"{data_name}/json_data_{time_name}"):
        os.mkdir(f"{data_name}/json_data_{time_name}")
    if not os.path.exists(f"{data_name}/json_data"):
        os.mkdir(f"{data_name}/json_data")
    for conv_id in struct_data:
        with open(f"{data_name}/json_data_{time_name}/{conv_id}.json", "w", encoding='utf-8') as f:
            json.dump(struct_data[conv_id], f, ensure_ascii=False, indent=4)
        with open(f"{data_name}/json_data/{conv_id}.json", "w", encoding='utf-8') as f:
            json.dump(struct_data[conv_id], f, ensure_ascii=False, indent=4)


def write_line_data():
    time_name = time_stamp().split(':')[0]
    if not os.path.exists(f"{data_name}/line_data_{time_name}"):
        os.mkdir(f"{data_name}/line_data_{time_name}")
    if not os.path.exists(f"{data_name}/line_data"):
        os.mkdir(f"{data_name}/line_data")
    for conv_id in line_data:
        with open(f"{data_name}/line_data_{time_name}/{conv_id}.txt", "w", encoding='utf-8') as f:
            for line in line_data[conv_id]:
                f.write(line + '\n')
        with open(f"{data_name}/line_data/{conv_id}.txt", "w", encoding='utf-8') as f:
            for line in line_data[conv_id]:
                f.write(line + '\n')


def form_topic():
    topic_pool = np.random.choice(topic_set, size=10)
    out = '<p>You are Apprentice!</p>Please chose one topic before the conversation!'
    for line in topic_pool:
        random_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        value = line.replace('"', ' _q_ ').replace("'", ' _q_ ')
        out += f'<p><input type="radio" id="{random_id}" name="topic" value="{value}" /><label for="{random_id}">{line}</label></p>'
    out += '<button onclick="topic_submit()" id="topic_submit" class="topic_submit">Submit</button>'
    return out


def form_knowledge(knowledge, score=None):
    out = ''
    keys = list(knowledge.keys())
    if score is not None:
        keys.sort(key=lambda x: max(score[x] + [0]), reverse=True)
    for title in keys[:7]:
        out += f'<p><b>{title}</b></p>'
        if score is not None and len(score[title]) > 0:
            tmp = [(j, v) for j, v in enumerate(knowledge[title])]
            tmp.sort(key=lambda x: score[title][x[0]], reverse=True)
            tmp = [x[1] for x in tmp]
            knowledge[title] = tmp
        for sent in knowledge[title][:10]:
            random_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            value = sent.replace('"', ' _q_ ').replace("'", ' _q_ ')
            out += f'<p><input type="radio" id="{random_id}" name="category" value="{value}" /><label for="{random_id}">{sent}</label></p>'
    return out


def safe_search(query, rows=5):
    try:
        return solr.search(query, rows=rows)
    except:
        return [{'title': ['Unknown ERROR. Please END this conversation. Very sorry!'],
                 'content': ['solr发生未知错误。请中止本次对话。十分抱歉']}]


def retrieval_topic(topic):
    topic = topic.lower()
    query = '&&'.join(
        list(set([x for x in ''.join(x if x.isalnum() else ' ' for x in topic).split()])))
    q = 'title:' + query

    knowledge = defaultdict(list)
    results = safe_search(q, rows=5)
    # results = solr.search(q, rows=5)
    title = [r['title'][0] for r in results]
    content = [r['content'][0] for r in results]
    for k, v in zip(title, content):
        knowledge[k].extend(sent_tokenize(v, language='english'))

    for k in knowledge:
        knowledge[k] = list(set(knowledge[k]))

    print(sum([len(v) for v in knowledge.values()]), topic, knowledge.keys())
    score = {k: [_f1_score(x, [topic]) for x in v] for k, v in knowledge.items()}
    return knowledge, score


def retrieval(dialogue, chosen_knowledge=None):
    dialogue = dialogue.lower()
    dialogue = translate(dialogue)
    dialogue = dialogue.lower()
    dialogue = '\n'.join(dialogue.split('\n')[-3:])
    dialogue = ' '.join(dialogue.split())
    print(f'query={dialogue}')

    dialogue = ' '.join([item[0] for item in nltk.pos_tag(dialogue.split()) if 'NN' in item[1]])

    query = '&&'.join(
        list(set([x for x in ''.join(x if x.isalnum() else ' ' for x in dialogue).split() if x not in STOPWORDS])))
    knowledge = defaultdict(list)
    q = 'title:' + query
    results = safe_search(q, rows=5)
    # results = solr.search(q, rows=5)
    title = [r['title'][0] for r in results]
    content = [r['content'][0] for r in results]
    for k, v in zip(title, content):
        knowledge[k].extend(sent_tokenize(v, language='english'))

    q = 'content:' + query
    results = safe_search(q, rows=5)
    # results = solr.search(q, rows=5)
    title = [r['title'][0] for r in results]
    content = [r['content'][0] for r in results]
    for k, v in zip(title, content):
        knowledge[k].extend(sent_tokenize(v, language='english'))

    for k in knowledge:
        knowledge[k] = list(set(knowledge[k]))

    for k in knowledge:
        if chosen_knowledge is not None:
            knowledge[k] = [x for x in set(knowledge[k]) if x not in chosen_knowledge]
        else:
            knowledge[k] = [x for x in set(knowledge[k])]
    print(sum([len(v) for v in knowledge.values()]), knowledge.keys())
    score = {k: [_f1_score(x, [dialogue]) for x in v] for k, v in knowledge.items()}
    return knowledge, score


def is_alive(username):
    return username in user_dict and user_dict[username] is not None and not user_dict[username].closed


def clean_user(username):
    user_dict.pop(username, 0)
    if username in waiting_list:
        waiting_list.pop(username, 0)
    if username in chatting_list:
        to_user = chatting_list[username]
        chatting_list.pop(username, 0)
        if is_alive(to_user):
            to_user_socket = user_dict.get(to_user)
            msg_dict = {'dialog_end': ''}
            to_user_socket.send(json.dumps(msg_dict))


def safe_send(username, msg_dict):
    if is_alive(username):
        user_socket = user_dict.get(username)
        user_socket.send(json.dumps(msg_dict))
    else:
        clean_user(username)


@app.route('/ws/<username>')
def ws_chat(username):
    print(f'\033[1;31m [{time_stamp()}] new login [{username}] \033[0m')
    user_socket = request.environ.get('wsgi.websocket')  # type:WebSocket
    user_dict[username] = user_socket
    while is_alive(username):
        msg = user_socket.receive()
        if msg is None:
            break
        msg_received = json.loads(msg)
        state = msg_received['state']
        if state == 'heart_beat':
            msg = {
                'heart_beat': 'heart_beat',
            }
            safe_send(username, msg)
        if state == 'match_request':
            request_user = msg_received['from_user']
            last_opponent = msg_received['last_opponent']
            if username in waiting_list:
                print(f"\033[1;33m [{username}] already in waiting list \033[0m")
            elif len(waiting_list) > 0:
                cache_list = deepcopy(waiting_list)
                for key in cache_list:
                    if key.split('.')[0] == request_user.split('.')[0]:
                        continue
                    if not is_alive(key):
                        continue
                    matched_user = key
                    waiting_list.pop(matched_user, 0)
                    last_id = int(conv_id_list[-1])
                    conv_id = str(last_id + 1)
                    conv_id_list.append(conv_id)

                    print(f"\033[1;33m [{conv_id}] [{username}] match with [{matched_user}] \033[0m")
                    data_saved[conv_id].append(f'[{time_stamp()}] conv_id={conv_id} start!')
                    line_data[conv_id].append(f'[{time_stamp()}] conv_id={conv_id},')
                    leaderboard[username.split('.')[0]].append(conv_id)
                    leaderboard[matched_user.split('.')[0]].append(conv_id)
                    struct_data[conv_id] = {'id': conv_id,
                                            'time_stamp': time_stamp(),
                                            'topic': '',
                                            'state': '',
                                            'role': dict(),
                                            'dialogue': [],
                                            'score': dict(),
                                            'current_knowledge': {'chosen_topic': None, 'retrieved_knowledge': None}}

                    wizard_msg = {
                        'conv_id': conv_id,
                        'knowledge': "You are Wizard! Apprentice is choosing topic, please wait.",
                        'role': 'Wizard',
                        'instruction': '你是Wizard。你的对手在选择topic,请等待对方选择完。当对方选择完,所选的topic将在左侧以及在对话框中显示。'
                    }
                    apprentice_msg = {
                        'conv_id': conv_id,
                        'knowledge': form_topic(),
                        'role': 'Apprentice',
                        'instruction': '你是Apprentice,请在左侧选择一个topic,点击submit。接下来的对话将从所选的话题展开！'
                    }
                    last_user = last_role.get(username.split('.')[0], 1)
                    last_matched = last_role.get(matched_user.split('.')[0], 1)
                    ratio = last_user / max(last_user + last_matched, 1)

                    if np.random.rand() > ratio:
                        wizard_msg['match'] = matched_user
                        wizard_msg['to_user'] = username
                        safe_send(username, wizard_msg)
                        apprentice_msg['match'] = username
                        apprentice_msg['to_user'] = matched_user
                        safe_send(matched_user, apprentice_msg)
                        data_saved[conv_id].append(f'[{time_stamp()}] [{username}] is Wizard')
                        data_saved[conv_id].append(f'[{time_stamp()}] [{matched_user}] is Apprentice')
                        line_data[conv_id][0] += f'wizard={username}, apprentice={matched_user}, '
                        struct_data[conv_id]['role'][username] = 'wizard'
                        struct_data[conv_id]['role'][matched_user] = 'apprentice'
                        last_role[username.split('.')[0]] = 1
                        last_role[matched_user.split('.')[0]] = 0
                        as_wizard[username.split('.')[0]].append(conv_id)
                        as_apprentice[matched_user.split('.')[0]].append(conv_id)
                    else:
                        apprentice_msg['match'] = matched_user
                        apprentice_msg['to_user'] = username
                        safe_send(username, apprentice_msg)
                        wizard_msg['match'] = username
                        wizard_msg['to_user'] = matched_user
                        safe_send(matched_user, wizard_msg)
                        data_saved[conv_id].append(f'[{time_stamp()}] [{username}] is Apprentice')
                        data_saved[conv_id].append(f'[{time_stamp()}] [{matched_user}] is Wizard')
                        line_data[conv_id][0] += f'wizard={matched_user}, apprentice={username}, '
                        struct_data[conv_id]['role'][matched_user] = 'wizard'
                        struct_data[conv_id]['role'][username] = 'apprentice'
                        last_role[username.split('.')[0]] = 0
                        last_role[matched_user.split('.')[0]] = 1
                        as_wizard[matched_user.split('.')[0]].append(conv_id)
                        as_apprentice[username.split('.')[0]].append(conv_id)
                    chatting_list[username] = matched_user
                    chatting_list[matched_user] = username
                    break
                else:
                    waiting_list[request_user] = last_opponent
                    print(f"\033[1;33m [{username}] is added to waiting list \033[0m")
            else:
                waiting_list[request_user] = last_opponent
                print(f"\033[1;33m [{username}] is added to waiting list \033[0m")
        elif state == 'dialog_end':
            if 'to_user' in msg_received:
                conv_id = msg_received['conv_id']
                to_user = msg_received['to_user']
                print(f"\033[1;33m [{conv_id}] [{username}] chose to end dialog with [{to_user}] \033[0m")
                data_saved[conv_id].append(f'[{time_stamp()}] [{username}] chose to end dialog with [{to_user}]')
                struct_data[conv_id]['state'] = 'chose to end'
                chatting_list.pop(username, 0)
                chatting_list.pop(to_user, 0)
                msg_dict = {'dialog_end': ''}
                safe_send(to_user, msg_dict)
            else:
                conv_id = msg_received['conv_id']
                print(f"\033[1;33m [{conv_id}] [{username}] confirm to end dialog \033[0m")
                data_saved[conv_id].append(f'[{time_stamp()}] [{username}] confirm to end dialog')
                struct_data[conv_id]['state'] = 'confirm to end'
                chatting_list.pop(username, 0)
                print(f"\033[1;33m [{conv_id}] dialog end \033[0m")
                data_saved[conv_id].append(f'[{time_stamp()}] dialog end !')
                struct_data[conv_id]['state'] = f'end at {time_stamp()}'
        elif state == 'score':
            conv_id = msg_received['conv_id']
            to_user = msg_received['to_user']
            score = msg_received['score']
            print(f"\033[1;35m [{conv_id}] [{username}] give [{to_user}] a [{score}] star ! \033[0m")
            data_saved[conv_id].append(f'[{time_stamp()}] [{conv_id}] [{username}] give [{to_user}] a [{score}] star !')
            struct_data[conv_id]['score'][to_user] = score
            starboard[to_user].append(score)
        elif state == 'topic':
            conv_id = msg_received['conv_id']
            to_user = msg_received['to_user']
            topic = msg_received['topic']
            msg_dict = {'topic': topic, 'from_user': msg_received['from_user'], 'to_user': to_user}
            chosen_knowledge, score = retrieval_topic(topic)
            msg_dict['knowledge'] = form_knowledge(chosen_knowledge, score) + no_passage_used
            print(f"\033[1;32m [{conv_id}] [{username}] chose topic [{topic}] \033[0m")
            data_saved[conv_id].append(f'[{time_stamp()}] [{conv_id}] [{username}] chose topic [{topic}]')
            struct_data[conv_id]['current_knowledge']['chosen_topic'] = deepcopy(chosen_knowledge)
            line_data[conv_id][0] += f'topic={topic}, '
            struct_data[conv_id]['topic'] = topic
            safe_send(to_user, msg_dict)
        elif state == 'message':
            conv_id = msg_received['conv_id']
            role = msg_received['role']
            topic = msg_received['topic']
            message = msg_received['chat']
            message = message.replace('\n', ' ').replace('\t', ' ')
            k = msg_received['knowledge']
            k = k.replace(' _q_ ', '"')
            to_user = msg_received['to_user']
            dialogue = msg_received['dialogue']
            msg_dict = {'chat': message, 'from_user': msg_received['from_user'], 'to_user': to_user}
            dialog_length[conv_id] += 1
            if len(message) == 0 or message == ' ' * len(message):
                if not is_alive(to_user):
                    clean_user(to_user)
                continue
            if role == 'Wizard':
                msg_dict['instruction'] = '你是Apprentice,请写出回复。尽可能让对话内容有趣！'
                print(
                    f"\033[1;32m [{conv_id}] [{username}] selected [{k}] and said [{message}] to [{to_user}] \033[0m")
                data_saved[conv_id].append(f'[{time_stamp()}] [{username}] selected [{k}] and said [{message}]')
                line_data[conv_id].append(f'[Wizard]: {message}')
                struct_data[conv_id]['dialogue'].append({
                    'time_stamp': time_stamp(),
                    'speaker': username,
                    'role': role,
                    'text': message,
                    'knowledge_pool': deepcopy(struct_data[conv_id]['current_knowledge']),
                    'select_knowledge': k
                })
            else:
                msg_dict['instruction'] = '你是Wizard,请在左侧选择一条知识,并基于这条知识写出回复。'
                print(f"\033[1;32m [{conv_id}] [{username}] said [{message}] to [{to_user}] \033[0m")
                data_saved[conv_id].append(f'[{time_stamp()}] [{username}] said [{message}]')
                line_data[conv_id].append(f'[Apprentice]: {message}')
                struct_data[conv_id]['dialogue'].append({
                    'time_stamp': time_stamp(),
                    'speaker': username,
                    'role': role,
                    'text': message
                })
                if topic != '':
                    chosen_topic, score = retrieval_topic(topic)
                    chosen_set = [line for item in chosen_topic.values() for line in item]
                    chosen_knowledge = form_knowledge(chosen_topic, score)
                else:
                    chosen_topic = {}
                    chosen_knowledge = ''
                    chosen_set = None
                retrieved_knowledge, score = retrieval(dialogue, chosen_knowledge=chosen_set)
                msg_dict['knowledge'] = chosen_knowledge + form_knowledge(retrieved_knowledge, score) + no_passage_used
                struct_data[conv_id]['current_knowledge']['chosen_topic'] = deepcopy(chosen_topic)
                struct_data[conv_id]['current_knowledge']['retrieved_knowledge'] = deepcopy(retrieved_knowledge)
                data_saved[conv_id].append(f'[{time_stamp()}] [{to_user}] saw knowledge [{msg_dict["knowledge"]}]')
                data_saved[conv_id].append(json.dumps(chosen_topic, ensure_ascii=False))
                data_saved[conv_id].append(json.dumps(retrieved_knowledge, ensure_ascii=False))
            safe_send(to_user, msg_dict)
    # time.sleep(10)
    # if not is_alive(username):
    #     clean_user(username)
    print(f'\033[1;31m [{time_stamp()}]', username, 'sign out ! websocket close', '\033[0m')
    return ''


@app.route('/chat')
def webchat():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    s = ''
    if len(leaderboard) > 0:
        x = list(leaderboard.keys())
        x.sort(key=lambda name: len([t for t in leaderboard[name] if dialog_length[t] >= 7]), reverse=True)
        for j, k in enumerate(x):
            s += f'<p>No.{j + 1} name={k}, num={len([t for t in leaderboard[k] if dialog_length[t] >= 7])}, as_wizard={len([t for t in as_wizard[k] if dialog_length[t] >= 7])}, as_apprentice={len([t for t in as_apprentice[k] if dialog_length[t] >= 7])}<p>'
    return f'<p>total_session_num={len(data_saved)}</p>' \
           f'<p>valid_session_num={len([t for t in dialog_length.values() if t >= 7])}</p>' \
           f'<p>chatting_user_num={len(chatting_list)}</p>' \
           f'<p>waiting_user_num={len(waiting_list)}</p>' \
           f'<p>active_user_num={len(user_dict)}</p>' \
           f'<p><b>leaderboard</b></p>' + s


def all_file(dirname):
    _f = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            _f.append(path)
    return _f


@app.route('/preview')
def preview():
    write_json()
    write_line_data()
    write_pkl(data_saved, f'./{data_name}/log.pkl')
    print(f'save data !')
    res = ''
    file_list = all_file(f'{data_name}/line_data')
    file_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]), reverse=True)
    for file in file_list:
        data = [line[:-1] for line in open(file, 'r', encoding='utf-8')]
        res += '<br>'.join(data)
        res += '<br>'
        res += '<br>'
    return res


if __name__ == '__main__':
    server = WSGIServer(('0.0.0.0', 9527), app, handler_class=WebSocketHandler)
    server.serve_forever()
