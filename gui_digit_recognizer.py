from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image, ImageOps
import numpy as np

# Load the model
model = load_model('mnist.h5')

def predict_digit(img):
    # Preprocess the image for the model
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    
    # Predict the digit
    res = model.predict([img])[0]
    prediction = np.argmax(res)
    confidence = np.max(res)
    
    return prediction, confidence

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack(side=tk.LEFT)
        
        self.label = tk.Label(self, text="Waiting...", font=("Helvetica", 24))
        self.label.pack(side=tk.RIGHT)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.update_prediction) 
        
        self.correction_entry = tk.Entry(self)
        self.correction_entry.pack(side=tk.BOTTOM)
        self.correct_btn = tk.Button(self, text="Correct", command=self.correct_prediction)
        self.correct_btn.pack(side=tk.BOTTOM)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(side=tk.BOTTOM)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.configure(text='Waiting...')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')

    def update_prediction(self, event=None):
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Get the canvas as an image
        img = ImageGrab.grab().crop((x, y, x1, y1))
        
        # Predict the digit
        prediction, confidence = predict_digit(img)
        
        # Update the label
        self.label.configure(text=f"Digit: {prediction}, Confidence: {confidence*100:.2f}%")

    def correct_prediction(self):
        correct_label = self.correction_entry.get()
        if correct_label.isdigit():
            x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
            y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            img = ImageGrab.grab().crop((x, y, x1, y1))
            self.save_feedback(img, correct_label)
        else:
            print("Please enter a valid digit.")

    def save_feedback(self, img, label):
        # Preprocess the image and save it with the label for retraining
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img = img.convert('L')
        img = ImageOps.invert(img)
        img.save(f'user_feedback_{label}.png')

app = App()
app.mainloop()
