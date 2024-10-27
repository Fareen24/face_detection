import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize a counter for saved images
image_save_count = 0

# Image Face detection 
def detect(image, scaleFactor=1.3, minNeighbors=5, color=(255, 0, 0)):
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)
    return image, faces

# Face detection function for webcam
def detect_faces_webcam(scaleFactor=1.3, minNeighbors=5, color=(255, 0, 0)):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video frames in Streamlit

    st.write("Starting webcam... please wait a few seconds.")
    time.sleep(3)  # Delay before starting face detection

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not access the webcam.")
            break

        # Detect faces in the frame
        processed_frame, faces = detect(frame, scaleFactor, minNeighbors, color)

        # Display frames in Streamlit
        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Stop when 'Stop Webcam' checkbox is unchecked
        if not st.checkbox("Stop Webcam", key="stop_webcam"):
            break

    cap.release()
    return processed_frame, faces  

# Streamlit app function
def app():
    global image_save_count  # Use global variable for image save count
    st.title("Face Detection App")
    st.write("Use options on the sidebar to start using the app.")
    
    # Instructions for the user
    st.write("""
    ### Instructions:
    1. **Select an option** from the sidebar.
    2. **Adjust Detection Parameters**: Use options on the sidebar to fine-tune face detection.
    3. **Save Detected Faces Image**: Optionally, you can save the detected face image on your device.
    """)

    activities = ["Home", "Upload a file and detect face", "Face detection using webcam"]
    choice = st.sidebar.selectbox("Select activity", activities)

    # Sidebar options for enhancements
    enhance_features = st.sidebar.radio('Enhance Features', 
                                        ['scaleFactor', 'minNeighbors', 'Pick rectangle color'])

    # Set default values
    scaleFactor = 1.3
    minNeighbors = 5
    color = (255, 0, 0)  # Default color is red

    if enhance_features == 'scaleFactor':
        scaleFactor = st.slider("Adjust Scale Factor", 1.0, 1.05, 1.1, 1.3)
        st.write(f"Scale Factor set to {scaleFactor}")

    elif enhance_features == 'minNeighbors':
        minNeighbors = st.slider("Adjust Min Neighbors", 1, 5, 3)
        st.write(f"Min Neighbors set to {minNeighbors}")

    elif enhance_features == 'Pick rectangle color':
        color = st.color_picker("Pick Rectangle Color", "#ff0000")
        color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        st.write(f"Selected Rectangle Color: {color}")

    if choice == "Home":
        image = Image.open('face_detection.webp')
        image = image.resize((400, 400))
        st.image(image, caption='An application by Chijo Nyacigak')
    
    elif choice == "Upload a file and detect face":
        st.subheader("Face Detection on an Image")

        image_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png', 'webp'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)
            
            # Convert the uploaded image to an array for OpenCV processing
            img_array = np.array(our_image.convert('RGB'))
            processed_image, faces = detect(img_array, scaleFactor, minNeighbors, color)

            st.image(processed_image, caption="Detected Faces")

            # Save button for the processed image
            if st.button("Save Detected Faces Image"):
                image_save_count += 1  # Increment the counter
                save_path = f"uploaded_face_detection_{image_save_count}.jpg"
                cv2.imwrite(save_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                st.success(f"Image saved as {save_path}")

    elif choice == "Face detection using webcam":
        st.subheader("Face Detection Using Webcam")
        if st.button("Start Webcam"):
            processed_frame, faces = detect_faces_webcam(scaleFactor=scaleFactor, minNeighbors=minNeighbors, color=color)
            
            # Option to save the last processed frame after stopping the webcam
            if st.button("Save Last Detected Frame"):
                if len(faces) > 0:
                    image_save_count += 1  
                    save_path = f"face_detected_{image_save_count}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    st.success(f"Image saved as {save_path}")
                else:
                    st.warning("No faces detected. Please try again.")

if __name__ == "__main__":
    app()
