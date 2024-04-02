# @app.route('/todo', methods=['POST'])
# def add_todo():
#     name = request.json['name']
#     is_executed = request.json['is_executed']

#     new_todo_item = TodoItem(name, is_executed)
#     db.session.add(new_todo_item)
#     db.session.commit()

#     return todo_schema.jsonify(new_todo_item)


# @app.route('/todo', methods=['GET'])
# def get_todos():
#     all_todos = TodoItem.query.all()
#     result = todos_schema.dump(all_todos)

#     return jsonify(result)


# @app.route('/todo/<id>', methods=['PUT', 'PATCH'])
# def execute_todo(id):
#     todo = TodoItem.query.get(id)

#     todo.is_executed = not todo.is_executed
#     db.session.commit()

#     return todo_schema.jsonify(todo)


# class TodoItem(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100))
#     is_executed = db.Column(db.Boolean)

#     def __init__(self, name, is_executed):
#         self.name = name
#         self.is_executed = is_executed


# # Todo schema
# class TodoSchema(ma.Schema):
#     class Meta:
#         fields = ('id', 'name', 'is_executed')


# # Initialize schema
# todo_schema = TodoSchema()
# todos_schema = TodoSchema(many=True)
