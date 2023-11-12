import base64
from datetime import datetime
import uuid
from flask import Flask, request, render_template, redirect, url_for
import os
from io import BytesIO
from solver import *

app = Flask(__name__)

PUZZLE_IMAGE_PATH = "puzzle2.png"
PUZZLE_DIMS = (6, 8)
UPLOADS_FOLDER = "static/uploads"

encoded_puzzle_pieces = setup_solver(PUZZLE_IMAGE_PATH, PUZZLE_DIMS)
    

def puzzle_solver_web(piece_image: IMG) -> tuple[IMG, IMG]:
    return solve_piece(piece_image, PUZZLE_IMAGE_PATH, PUZZLE_DIMS, encoded_puzzle_pieces)

def encode_image(image: IMG, prefix: str, session_id: str) -> str:
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # save the image to a file in the uploads folder / YYYY-MM-DD-HH / prefix_UUID.jpg
    date_time = datetime.now().strftime("%Y-%m-%d-%H")
    path = os.path.join(UPLOADS_FOLDER, date_time, f"{prefix}_{session_id}.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path    
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if not file.filename:
            return redirect(request.url)
        if file:
            # Generate a unique identifier for this session
            session_id = str(uuid.uuid4())
            
            # Convert the uploaded file to a PIL Image
            input_image = Image.open(file.stream)

            # Solve the puzzle
            output_image, cropped_input_image = puzzle_solver_web(input_image)

            output_path = encode_image(output_image, "output", session_id).replace("\\", "/").replace("static/", "")
            input_path = encode_image(cropped_input_image, "input", session_id).replace("\\", "/").replace("static/", "")

            # Redirect to display the processed image
            return redirect(url_for('display_image', output_file=output_path, input_file=input_path))

    return render_template('upload.html')

@app.route('/display_image')
def display_image():
    input_file = request.args.get('input_file', '')
    output_file = request.args.get('output_file', '')
    return render_template('display_image.html', 
                            input_file=input_file, 
                            output_file=output_file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
