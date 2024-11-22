from flask import Flask, render_template, request, jsonify
import importlib.util
import os
import sys
import json
import numpy as np

# Расчёт пути к родительской директории
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Проверка наличия родительской директории в sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Безопасный импорт модуля
SLON_name = 'main_model\\main'
spec_SLON = importlib.util.spec_from_file_location(SLON_name, os.path.join(parent_dir, f"{SLON_name}.py"))
SLON = importlib.util.module_from_spec(spec_SLON)
spec_SLON.loader.exec_module(SLON)


app = Flask(__name__)

app.static_folder = 'static'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form
    try:
        result = process_answers(data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})


def process_answers(answers):
    answers = list(dict(answers).values())
    if 26 < len(answers) < 26: return "Ты уебан"

    with open(parent_dir + '\\web\\static\\RU2ENG.json', 'r', encoding='utf-8') as file:
        RU2ENG = json.load(file)
    keys = RU2ENG.keys()
    for x in answers:
        if x in keys:
            answers[answers.index(x)] = RU2ENG[x]
        
    with open(parent_dir + '\\web\\static\\ENG2int.json', 'r', encoding='utf-8') as file:
        ENG2int = json.load(file)
    keys = ENG2int.keys()
    for x in answers:
        if x in keys:
            answers[answers.index(x)] = ENG2int[x]

    for x in answers:
        try:
            val = int(x)
        except ValueError:
            return "Ты ебанидзе"
        
    answers = np.array(answers, dtype=int) + 1
    print(answers)
    result = SLON.SLON(answers)
    print(result)
    return result


if __name__ == '__main__':
    app.run(debug=True)