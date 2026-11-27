from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import base64


STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/test", methods = ['GET'])
def test():
    return "Test request received seccessfully. Service is running"


@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('landing.html')


@app.route('/predict', methods = ['POST'])
def predict():
    predictor = pkl.load(open(r'xgb_classifier.pkl','rb'))
    scaler = pkl.load(open(r'scaler.pkl','rb'))
    cv = pkl.load(open(r'countVectorizer.pkl','rb'))

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        
        if 'file' in request.files:

            # Bulk prediction from csv file
            file = request.files['file']   #fetches the uploaded CSV.
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers['X-Graph-Exists'] = 'true'
            response.headers['X-Graph-Data'] = base64.b64encode(graph.getbuffer()).decode('ascii')

            return response

        elif "text" in request.json:
            # Single string prediction 
            text_input = request.json['text']
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})
        
    except Exception as e:
        return jsonify({"error": str(e)})
    

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)

    x_prediction = cv.transform(corpus).toarray()
    x_prediction_scale = scaler.transform(x_prediction)

    # y_prediction = predictor.predict_proba(x_prediction_scale)
    # y_prediction = y_prediction.argmax(axis=1)[0]

    y_prediction = predictor.predict(x_prediction_scale)[0]

    return "Positive" if y_prediction == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    x_prediction = cv.transform(corpus).toarray()
    x_prediction_scale = scaler.transform(x_prediction)

    # y_prediction = predictor.predict_proba(x_prediction_scale)
    # y_prediction = list(map(sentiment_mapping, y_prediction))

    y_prediction = predictor.predict(x_prediction_scale)
    y_prediction = ["Positive" if x == 1 else "Negative" for x in y_prediction]


    data["Predicted sentiment"] = y_prediction
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index = False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize = (5,5))
    colors = ('green', 'red')
    wp = {'linewidth': 1, "edgecolor": 'black'}
    tags = data['Predicted sentiment'].value_counts()
    explode = (0.01, 0.01)

    tags.plot(kind = 'pie', autopct = "%1.1f%%", shadow = True, colors = colors, explode = explode, wedgeprops = wp, startangle = 90, title = "Sentiment Distribution", xlabel = "", ylabel = "")

    graph = BytesIO()
    plt.savefig(graph, format = 'png')
    plt.close()


    return graph

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    app.run(port = 5000, debug = True)      
