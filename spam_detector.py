# Spam Message Detector - GUI VERSION 
# Using Naive Bayes + TF-IDF + Tkinter UI
# Author: koustum vijayant

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. DATASET
data = {
    "message": [
        "Congratulations! You won a free lottery ticket.",
        "Hello, how are you?",
        "Claim your free prize now!",
        "Are we meeting tomorrow?",
        "Please call me later.",
        "Win a brand new car! Limited offer.",
        "Let's have lunch today.",
        "Urgent! Your account has been compromised.",
        "Send me your assignment."
    ],  
    "label": ["spam", "ham", "spam", "ham", "ham", "spam", "ham", "spam", "ham"]
}
df = pd.DataFrame(data)

# 2. CLEAN TEXT FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text)          # Collapse extra whitespace
    return text.strip()

df["cleaned"] = df["message"].apply(clean_text)

# 3. MODEL TRAINING
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(df["cleaned"])

model = MultinomialNB()
model.fit(X_vec, df["label"])

# 4. GUI SETUP
root = tk.Tk()
root.title("Spam Message Detector - AI & ML Project")
root.geometry("500x400")
root.config(bg="#1B1B1B")

title_label = tk.Label(
    root,
    text="Spam Message Detector",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#1B1B1B"
)
title_label.pack(pady=20)

msg_label = tk.Label(
    root,
    text="Enter your message:",
    font=("Arial", 12),
    fg="white",
    bg="#1B1B1B"
)
msg_label.pack()

msg_box = tk.Text(root, height=5, width=50, font=("Arial", 12))
msg_box.pack(pady=10)

# DETECT SPAM FUNCTION
def detect_spam_gui():
    message = msg_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Error", "Please enter a message.")
        return

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]

    if result == "spam":
        messagebox.showerror("Result", "🚨 This message is SPAM!")
    else:
        messagebox.showinfo("Result", "✅ This message is NOT spam.")

# CLEAR FUNCTION
def clear_text():
    msg_box.delete("1.0", tk.END)

# KEYBOARD BINDING (Enter Key Triggers Detection)
def on_enter_key(event):
    detect_spam_gui()
    return "break"  # Prevents adding a new line in the text box

msg_box.bind("<Return>", on_enter_key)

# 5. BUTTON LAYOUT
btn_frame = tk.Frame(root, bg="#1B1B1B")
btn_frame.pack(pady=20)

detect_button = tk.Button(
    btn_frame,
    text="Detect Spam",
    width=15,
    height=2,
    command=detect_spam_gui
)
detect_button.grid(row=0, column=0, padx=10)  # Placed in column 0

clear_button = tk.Button(
    btn_frame,
    text="Clear",
    width=15,
    height=2,
    command=clear_text
)    
clear_button.grid(row=0, column=1, padx=10)  # Placed in column 1

root.mainloop()