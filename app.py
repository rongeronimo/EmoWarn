from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

#fake news classifier dependencies
fake_news_model = pickle.load(open("fake_classifier.pkl", "rb"))
fake_vectorizer = pickle.load(open("fake_vectorizer.pkl", "rb"))
fake_transformer = pickle.load(open("fake_transformer.pkl", "rb"))

#emotion classifier dependencies
emotion_model = pickle.load(open("emo_classifier.pkl", "rb"))
emotion_vectorizer = pickle.load(open("emo_vectorizer.pkl", "rb"))
emotion_transformer = pickle.load(open("emo_transformer.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    #grab user input from form
    raw_input = request.form.get('input')
    to_process = [raw_input]
    to_process_emo = [raw_input]

    #preprocess data to fit into model
    to_process = fake_vectorizer.transform(to_process)
    to_process = fake_transformer.transform(to_process)

    #predict using loaded pickled model
    prediction = fake_news_model.predict(to_process)

    # preprocess data to fit into model
    to_process_emo = emotion_vectorizer.transform(to_process_emo)
    to_process_emo = emotion_transformer.transform(to_process_emo)

    # predict using loaded pickled model
    emo_prediction = emotion_model.predict(to_process_emo)

    #return render_template("index.html",prediction_text="The fake news classification result is {}".format(prediction))
    return render_template("index.html", prediction_text="The fake news classification result is "+prediction +
                                                        " and the emotion classification result is "+emo_prediction)


if __name__ == "__main__":
    flask_app.run(debug=True)
