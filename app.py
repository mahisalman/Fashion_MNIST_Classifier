from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
model = load_model("best_cnn_model.keras")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Ankle boot', 'Bag']

import numpy as np
from PIL import ImageOps, Image
from scipy.ndimage import center_of_mass, shift

def preprocess_image(image):
    from PIL import ImageOps
    import numpy as np
    from scipy.ndimage import center_of_mass, shift

    # Step 1: Convert to grayscale
    image = image.convert("L")

    # Step 2: Resize with high-quality resampling while keeping aspect ratio
    image = ImageOps.invert(image)  # Invert first to better detect clothing
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Step 3: Convert to NumPy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Step 4: Threshold noise (optional, helps sharpen low-contrast scans)
    img_array[img_array < 0.2] = 0.0

    # Step 5: Center mass of the foreground (white pixels)
    cy, cx = center_of_mass(img_array)
    if np.isnan(cx) or np.isnan(cy):  # Handle empty or black image
        cx, cy = 14, 14
    shift_y = np.round(14 - cy).astype(int)
    shift_x = np.round(14 - cx).astype(int)

    # Step 6: Shift the image to center the shape
    img_array = shift(img_array, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Step 7: Reshape for CNN input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_data = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            processed = preprocess_image(image)

            # Get prediction
            preds = model.predict(processed)[0]
            top_indices = preds.argsort()[-3:][::-1]
            prediction = [
                class_names[top_indices[0]], round(preds[top_indices[0]] * 100, 2),
                class_names[top_indices[1]], round(preds[top_indices[1]] * 100, 2),
                class_names[top_indices[2]], round(preds[top_indices[2]] * 100, 2),
            ]

            # Convert uploaded image to base64 for persistent preview
            buffered = BytesIO()
            image.convert("RGB").save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

    return render_template('index.html', prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
