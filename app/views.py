# Important imports
from app import app
from flask import request, render_template
import cv2
import numpy as np
from PIL import Image
import string
import random
import os
from retinaface import RetinaFace

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

def get_array_img(img):
    image = Image.open(img)
    image = image.resize((300, 300))
    image_arr = np.array(image)
    return image_arr

def align_face(img, landmarks):
    """
    Align face using 5-point RetinaFace landmarks.
    landmarks = {'left_eye': (x,y), 'right_eye': (x,y), 'nose': (x,y),
                 'mouth_left': (x,y), 'mouth_right': (x,y)}
    """
    # reference 5 points for alignment
    ref_points = np.float32([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # mouth left
        [70.7299, 92.2041]    # mouth right
    ])
    ref_size = (112, 112)

    src_points = np.float32([
        landmarks["left_eye"],
        landmarks["right_eye"],
        landmarks["nose"],
        landmarks["mouth_left"],
        landmarks["mouth_right"]
    ])

    # estimate similarity transform
    M = cv2.estimateAffinePartial2D(src_points, ref_points, method=cv2.LMEDS)[0]
    aligned_face = cv2.warpAffine(img, M, ref_size, borderValue=0.0)
    return aligned_face

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        full_filename = 'images/white_bg.jpg'
        return render_template("index.html", source=full_filename, destination=full_filename, result=full_filename)

    if request.method == "POST":
        source_upload = request.files['source_image']
        destination_upload = request.files['destination_image']

        # generating unique names
        letters = string.ascii_lowercase
        sname = ''.join(random.choice(letters) for i in range(10)) + '.png'
        sfull_filename = 'uploads/' + sname
        dname = ''.join(random.choice(letters) for i in range(10)) + '.png'
        dfull_filename = 'uploads/' + dname
        rname = ''.join(random.choice(letters) for i in range(10)) + '.png'
        rfull_filename = 'uploads/' + rname

        simg = Image.open(source_upload)
        simg.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], sname))
        dimg = Image.open(destination_upload)
        dimg.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], dname))

        img = get_array_img(source_upload)
        img2 = get_array_img(destination_upload)

        # Detect faces with landmarks
        faces1 = RetinaFace.detect_faces(img)
        faces2 = RetinaFace.detect_faces(img2)

        if not isinstance(faces1, dict) or len(faces1) == 0:
            return "No face detected in source image"
        if not isinstance(faces2, dict) or len(faces2) == 0:
            return "No face detected in destination image"

        # take first face only
        face1_key = list(faces1.keys())[0]
        face2_key = list(faces2.keys())[0]

        face1_info = faces1[face1_key]
        face2_info = faces2[face2_key]

        # Get landmarks
        landmarks1 = face1_info["landmarks"]
        landmarks2 = face2_info["landmarks"]

        # Align faces
        aligned_face1 = align_face(img, landmarks1)
        aligned_face2 = align_face(img2, landmarks2)

        # Resize source face to destination face size
        face_h, face_w, _ = aligned_face2.shape
        face1_resized = cv2.resize(aligned_face1, (face_w, face_h))

        # Create mask for blending
        face1_gray = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(face1_gray, 1, 255, cv2.THRESH_BINARY)

        # Find center of destination face bbox
        x, y, w, h = face2_info["facial_area"]
        center = (x + w // 2, y + h // 2)

        # Seamless cloning
        output = cv2.seamlessClone(face1_resized, img2, mask, center, cv2.NORMAL_CLONE)

        resultimg = Image.fromarray(output, 'RGB')
        resultimg.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], rname))

        return render_template('index.html', source=sfull_filename, destination=dfull_filename, result=rfull_filename)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
