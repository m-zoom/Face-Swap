# Magical Face Swap Tool 😄😄 

## What is this?

This is a fun tool that lets you swap your face with another face! When you look into your webcam, you'll see your face change to look like someone else. It's like wearing a mask, but it's digital!


##  What You Need Before Starting

- A computer with a webcam
- Python installed on your computer
- A picture of the face you want to swap with (save it as "reference_face2.jpg")

## 📋 Step-by-Step Installation Guide

### Step 1: Install Python

If you don't have Python yet:
1. Go to [python.org](https://www.python.org/downloads/)
2. Click the big "Download Python" button
3. Follow the installation steps (remember to check "Add Python to PATH" during installation)

### Step 2: Install Required Packages

1. Open your computer's command prompt or terminal
2. Copy and paste these commands, pressing Enter after each one:

```
pip install opencv-python
pip install numpy
pip install mediapipe
```

### Step 3: Get the Face Swap Program

1. Download this program and save it as `face_swapper.py`
2. Put your reference face image in the same folder as the program
3. Name your reference image `reference_face2.jpg`

## 🚀 How to Use the Face Swap Tool

### Starting the Program

1. Open your command prompt or terminal
2. Navigate to the folder where you saved the program (use the `cd` command)
3. Type `python face_swapper.py` and press Enter

### Using the Face Swap

1. The program will automatically detect when your webcam is on
2. Two windows will appear side by side:
   - Left side: Your normal webcam view
   - Right side: Your face with the swap applied
3. To exit the program, press the 'q' key on your keyboard

## 🤔 Common Questions

### What if it can't find my reference image?

If the program can't find your image, it will ask you to type in the full path to the image. This means you need to tell the program exactly where your image is stored on your computer.

Example: `C:\Users\YourName\Pictures\reference_face2.jpg`

### What if no face appears on the right side?

Make sure:
1. Your face is clearly visible to the webcam
2. There's enough light in your room
3. The reference image has a clearly visible face

### What if the program doesn't start?

Make sure:
1. You installed all the required packages
2. Your webcam is working properly
3. No other program is currently using your webcam

## 💡 How It Works (For Curious Minds)

This program uses a technology called "facial landmarks" to find the important parts of your face (like eyes, nose, mouth). Then it carefully takes the corresponding parts from the reference face and places them onto your face in real-time. The program does this so quickly that it looks like magic!

## 🛠️ Advanced Options

If you want to use a different reference image:
1. Put your new image in the same folder as the program
2. When running the program, it will ask for the path to your image if it can't find the default one
3. Type the name of your new image (if it's in the same folder) or the full path to the image

## 🆘 Need Help?

If you're having trouble:
1. Make sure your webcam is working with other applications
2. Check that you've installed all the required packages
3. Try using a different reference image with a clear, front-facing face

Enjoy your magical face transformation! 😄
