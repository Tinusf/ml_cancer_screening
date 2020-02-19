from flask import Flask, request
import json
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import numpy
from io import BytesIO
import main

app = Flask(__name__)
CORS(app)

model = main.get_saved_model()


def base64_to_numpy(base64string):
    dimensions = (28, 28)

    encoded_image = base64string.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image)).convert('RGB')
    # image is (28, 28)
    img = img.resize(dimensions, Image.ANTIALIAS)

    pixels = numpy.asarray(img, dtype='uint8')
    return pixels


def format_output(predicted):
    return {
        "Actinic Keratoses": float(predicted[0]),
        "Basal cell carcinoma": float(predicted[1]),
        "Benign keratosis": float(predicted[2]),
        "Dermatofibroma": float(predicted[3]),
        "Melanocytic nevi": float(predicted[4]),
        "Melanoma": float(predicted[5]),
        "Vascular skin lesions": float(predicted[6])
    }


@app.route('/', methods=["POST"])
def check_image():
    json_input_data = json.loads(request.data)
    pixels = base64_to_numpy(json_input_data["data"])
    predicted = main.predict(model, np.array([pixels]))

    return json.dumps(format_output(predicted))
