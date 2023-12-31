from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import io

app = Flask(__name__)

# Load the pre-trained JSON model
json_file = open('C:/Users/akshatvijay4/Desktop/Noteify/resnet_50_model.json', 'r')

#reading model
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/Users/akshatvijay4/Desktop/Noteify/currency_detector_model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
class_labels = [
    '10','100','20','200','2000','50','500','Background'
]
from tensorflow.keras.preprocessing import image

def prediction(file_name):
    img = image.load_img(file_name, target_size=(256,256))

    image_to_test = image.img_to_array(img)

    #since Keras expects a list of images, not a single image,
    # Add a fourth dimension to the image

    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the model
    results = loaded_model.predict(list_of_images)

    # Since we are only testing one image, we only need to check the first result
    single_result = results[0]

    # We will get a likelihood score for all  possible classes.
    # Find out which class had the highest score.
    # the class with highest likelihood is predicted as the result.

    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]

    # Get the name of the most likely class
    class_label = class_labels[most_likely_class_index]

    # Print the result
    print(file_name)
    return("This is image of a {} - Likelihood: {: .2f}".format(class_label, class_likelihood))

@app.route('/', methods=['GET', 'POST'])
def index():
    detected_currency = None

    if request.method == 'POST':
        image = request.files['image']

        if image:
            image_data = image.read()
            detected_currency = prediction(io.BytesIO(image_data))

    return render_template('index.html', detected_currency=detected_currency)

if __name__ == '__main__':
    app.run(debug=True)
