import os
import time
import tempfile
from PIL import Image
import streamlit as st
from ultralytics import YOLO 
from datetime import datetime, timedelta 

detect_dir = "C:/Users/USER_NAME/runs/detect"

st.write('# Helmet Detection & Traffic Signal Control: Enhancing Safety on the Road')
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

def draw_traffic_signal(color):

    # Draw the circles representing the traffic lights
    st.sidebar.markdown('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="width: 50px; height: 50px; border-radius: 50%; background-color: {color}; margin: 5px;"></div>', unsafe_allow_html=True) # Traffic light colour
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            images.append(image)
    return images

def resize_images_in_directory(directory, target_width, target_height):
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            img = Image.open(image_path)
            img_resized = img.resize((target_width, target_height))
            img_resized.save(image_path)  # Overwrite the original image with the resized one

def show_traffic_signal(status):
    st.sidebar.empty()
    if status == "running":
        draw_traffic_signal("red")
    elif status == "halfway":
        draw_traffic_signal("yellow")
    elif status == "ended":
        draw_traffic_signal("green")

if uploaded_file is not None:
    st.write(uploaded_file.type)

    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getvalue())

    # Display the uploaded image
    uploaded_image = Image.open(temp_file_path)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    # st.write(f"Uploaded image shape: {np.array(uploaded_image).shape}")

    if uploaded_file.type.startswith('image'):
        
        #best_h.pt has classes[0:with helmet 1:without helmet]
        model = YOLO("best50_h.pt")
        results = model.predict(source=uploaded_image, save_crop=True, conf=0.25, classes=[1])

        predict_folders1 = [folder for folder in os.listdir(detect_dir) if folder.startswith('predict')]
        latest_predict_folder1 = max(predict_folders1, key=lambda folder: os.path.getctime(os.path.join(detect_dir, folder)))
        predict_dir1 = os.path.join(detect_dir, latest_predict_folder1)
        subdirectory = "crops/Without Helmet"
        images_directory = os.path.join(predict_dir1,subdirectory)

        target_width = 150
        target_height = 150
        resize_images_in_directory(images_directory, target_width, target_height)
        
        # Load images from directory
        images = load_images_from_directory(images_directory)

        if images:
            st.header("Images")
            st.image(images, width=150)

            # Calculate the timer duration based on the number of images
            timer_duration = 30 + len(images) * 5
            
            # Display the number of persons detected without a helmet
            st.write(f"Number of persons detected without helmet: {len(images)}")

            # Display the timer
            st.write(f"Timer: {timer_duration} seconds")

            # Display the timer with increased text size
            timer_placeholder = st.empty()
            end_time = datetime.now() + timedelta(seconds=timer_duration)
            halfway_time = end_time - timedelta(seconds=timer_duration // 2)
            count_y=0 
            show_traffic_signal("running")
            while datetime.now() < end_time:
                remaining_time = end_time - datetime.now()
                if datetime.now() >= halfway_time and count_y==0:
                    show_traffic_signal("halfway")  # Update traffic signal status to halfway
                    count_y=1
                timer_placeholder.markdown(f"<h2>Time remaining: {remaining_time.seconds} seconds</h2>", unsafe_allow_html=True)
                time.sleep(1)    
            show_traffic_signal("ended")
            st.write("Timer has ended!")
        else:
            #("No detection of persons without helmet.")
            # Calculate the timer duration based on the number of images
            timer_duration = 30 

            # Display the timer
            timer_placeholder = st.empty()
            end_time = datetime.now() + timedelta(seconds=timer_duration)
            halfway_time = end_time - timedelta(seconds=timer_duration // 2)
            count_y=0
            while datetime.now() < end_time:
                remaining_time = end_time - datetime.now()
                if datetime.now() >= halfway_time and count_y==0:
                    show_traffic_signal("halfway")  # Update traffic signal status to halfway
                    count_y=1
                timer_placeholder.markdown(f"<h2>Time remaining: {remaining_time.seconds} seconds</h2>", unsafe_allow_html=True)
                time.sleep(1)
            st.write("Timer has ended!")
            