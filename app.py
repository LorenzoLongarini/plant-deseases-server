from flask import  Flask, request, jsonify
from utils import init, do_query
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
vectordb, chain = init()


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')


# suppress SQLALCHEMY_TRACK_MODIFICATIONS warning
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

db_path = './db.sqlite'

class LlmItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(100))

    def __init__(self, answer):
        self.answer = answer


# Todo schema
class LlmSchema(ma.Schema):
    class Meta:
        fields = ('id', 'answer')


# Initialize schema
llm_schema = LlmSchema()
llms_schema = LlmSchema(many=True)


@app.route('/llm/<id>', methods=['DELETE'])
def delete_answer(id):
    answer_to_delete = LlmItem.query.get(id)
    db.session.delete(answer_to_delete)
    db.session.commit()

    return llm_schema.jsonify(answer_to_delete)

@app.route('/', methods=['GET'])
def hello_world():
    return 'hello'

@app.route('/llm', methods=['POST'])
def add_question():
    query = request.json['query']
    # query = "Tell me in 10 words what is bitbattle."
    context, answer = do_query(vectordb, chain, query)
    new_llm_item = LlmItem(answer)
    db.session.add(new_llm_item)
    db.session.commit()

    return llm_schema.jsonify(new_llm_item)


@app.route('/llm', methods=['GET'])
def get_questions():
    all_llms = LlmItem.query.order_by(LlmItem.id.desc()).first()
    result = llm_schema.dump(all_llms)

    return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()

    vectordb, chain = init()

    # app.run(debug=True)
    app.run(host="localhost", port=8000, debug=True)