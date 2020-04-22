from flask import Flask, request
import json
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import numpy
from io import BytesIO
from SkinCancerClassifier import SkinCancerClassifier

# Create a new flask app
app = Flask(__name__)
# Enable CORS for this API
CORS(app)

# Create a classifier instance.
skin_cancer_classifier = SkinCancerClassifier()


def base64_to_numpy(base64string):
    """
    :param base64string: The strang that should be decoded.
    :return: A numpy array containing the image.
    """
    # Dimensions of the image
    dimensions = (28, 28)

    encoded_image = base64string.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image)).convert('RGB')
    # image is (28, 28)
    img = img.resize(dimensions, Image.ANTIALIAS)

    pixels = numpy.asarray(img, dtype='float64')
    return pixels


def format_output(predicted):
    """
    :param predicted: An array containing the probabilities of teach class.
    :return: Formatted dictionary with the names of each class as keys and the probabilies as
    values.
    """
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
    # Load the payload with json.
    json_input_data = json.loads(request.data)
    # Convert base64 data to numpy array.
    pixels = base64_to_numpy(json_input_data["data"])
    # Get the predicted classes.
    predicted = skin_cancer_classifier.model.predict(np.array([pixels]))
    # Format the output.
    return json.dumps(format_output(predicted))
