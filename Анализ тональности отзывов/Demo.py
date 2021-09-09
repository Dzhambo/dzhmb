from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length
from sentiment_classifier import SentimentClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret_key"
bootstrap = Bootstrap(app)
print("Загрузка классификатора...")
classifier = SentimentClassifier()
print("Классификатор загружен")


class NameForm(FlaskForm):
    """Класс для отображения формы"""
    name = TextAreaField("Отзыв", validators=[DataRequired(), Length(1, 5000)])
    submit = SubmitField('Оценить')


@app.route('/', methods=['POST','GET'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        name = classifier.get_prediction_message(name)
    return render_template('hello.html', form=form, name=name)


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=False, port=8080)

