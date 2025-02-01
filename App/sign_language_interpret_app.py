#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import threading

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the correct input size for fc1
        self._initialize_fc1()

        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _initialize_fc1(self):
        dummy_input = torch.zeros(1, 3, 200, 200)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        self.fc1_input_size = x.size(1)
        print(f"Calculated fc1 input size: {self.fc1_input_size}")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor dynamically
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()

model = torch.load('/pgarg2_shreyath_ppatil22_final_projectt_simple_cnn_model.pth', map_location=torch.device('cpu'))
model.eval()

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def capture_frames(stop_event):
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        time.sleep(10)  # Wait for 10 seconds before capturing the next frame
    cap.release()

# Function to preprocess a single frame
def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = preprocess(img)
    return img_t

# Function to classify a single frame
def classify_frame(frame):
    frame_t = preprocess_frame(frame)
    frame_t = frame_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(frame_t)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to generate text from predictions
def generate_text(predictions):
    # Implement your logic to combine predictions into words
    text = ''.join([chr(65 + pred) for pred in predictions])  # Example: convert class indices to letters
    return text


# Streamlit UI
st.title("Sign Language to Text")

if 'capturing' not in st.session_state:
    st.session_state.capturing = False

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None

def start_capture():
    st.session_state.capturing = True
    st.session_state.stop_event = threading.Event()
    capture_images()

def stop_capture():
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.capturing = False

def capture_images():
    text_placeholder = st.empty()
    image_placeholder = st.empty()
    for frame in capture_frames(st.session_state.stop_event):
        image_placeholder.image(frame, channels="BGR")
        pred = classify_frame(frame)
        st.session_state.predictions.append(pred)
        current_text = generate_text(st.session_state.predictions)
        text_placeholder.text(f"Current Text: {current_text}")
        #suggestions = suggest_words(current_text)
        #st.write(f"Suggestions: {suggestions}")
        if not st.session_state.capturing:
            break

if st.button('Start Capture'):
    if not st.session_state.capturing:
        start_capture()

if st.button('Stop Capture'):
    stop_capture()

if st.button('Clear'):
    st.session_state.predictions = []
    st.write("Cleared")
