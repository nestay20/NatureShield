import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image
import csv
import io

#=====================================Load labels, infer, and preprocess=====================================#

# Load CSV labels for names
def load_labels(label_path):
    for enc in ('utf-8-sig', 'utf-16'):
        try:
            with io.open(label_path, 'r', encoding=enc) as f:
                reader = csv.reader(f)
                labels = {}
                for row in reader:
                    if not row:
                        continue
                    try:
                        idx = int(row[0])
                    except ValueError:
                        continue
                    if len(row) >= 3:
                        sci, com = row[1], row[2]
                    else:
                        sci, com = row[1], ''
                    labels[idx] = (sci, com)
            return labels
        except UnicodeError:
            continue
    return labels

# Loads the image in the models dtype 
def preprocess(image_path, input_size, interpreter):
    # Load image from disk 
    img = Image.open(image_path).convert("RGB")
    # Crop the image to the center
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    # Resize to match models dimensions
    img = img.resize(input_size, Image.BILINEAR)
    # Convert the image to NumPy array
    arr = np.asarray(img, dtype=np.float32)
    # Not sure how this works took it from tensorflow docs
    inp_det = interpreter.get_input_details()[0]
    dtype = inp_det["dtype"]
    scale, zp = inp_det["quantization"]
    if dtype == np.uint8:
        arr = (arr / 255.0 / scale + zp).round().clip(0, 255).astype(np.uint8)
    else:
        arr = (arr / 255.0).astype(np.float32)
    return np.expand_dims(arr, axis=0)


def infer(interpreter, input_data):
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp_det["index"], input_data)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_det["index"])[0]
    idx = int(np.argmax(raw))
    return idx

#=====================================UI=====================================#

# Map for bubble selection
MODEL_CONFIG = {
    "Plants":  ("models/plants/model.tflite",  "models/plants/labels.csv"),
    "Birds":   ("models/birds/model.tflite",   "models/birds/labels.csv"),
    "Insects": ("models/insects/model.tflite", "models/insects/labels.csv"),
    "Mammals": ("models/mammals/model.tflite", "models/mammals/labels.csv"),
}

class NatureShieldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSCI 436/536 NatureShield")
        self.geometry("400x240")
        self.resizable(False, False)

        tk.Label(self, text="CSCI 436/536 NatureShield", font=("Arial", 16, "bold")).pack(pady=10)

        # Category buttons so we know what model to use
        self.category = tk.StringVar(value="Plants")
        frame = tk.Frame(self)
        for cat in MODEL_CONFIG:
            tk.Radiobutton(frame, text=cat, variable=self.category, value=cat).pack(side="left", padx=10)
        frame.pack(pady=5)

        # Button to allow uploading image from device
        tk.Button(self, text="Upload Image", command=self.upload).pack(pady=10)

        # Label for when we need to print species name
        self.result_lbl = tk.Label(self, font=("Arial", 12))
        self.result_lbl.pack(pady=20)

    def upload(self):
        # Prompt image formats supported
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        cat = self.category.get()
        model_path, labels_path = MODEL_CONFIG[cat]
        # Preprocess and infer
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        _, H, W, _ = interpreter.get_input_details()[0]["shape"]
        idx = infer(
            interpreter,
            preprocess(path, (W, H), interpreter)
        )

        # Get the identified species scientific and common name and display it
        labels = load_labels(labels_path)
        sci, com = labels.get(idx, ("Unknown", "Unknown"))
        display = f"{com} ({sci})"
        self.result_lbl.config(text=f"{display}")

if __name__ == "__main__":
    app = NatureShieldApp()
    app.mainloop()
