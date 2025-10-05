# 🎨 Air Drawing – Virtual Paint with Hand Gestures  

This project is a **hand gesture-controlled virtual drawing app** built using **OpenCV**, **MediaPipe**, and **NumPy**.  
It allows you to draw in the air using your **index finger**, switch colors using **simple hand gestures**, and even clear the canvas — all without touching your keyboard or mouse!  

---

## 🚀 Features  
- 🖌️ Draw in the air with your index finger  
- 🌈 Switch brush colors using gestures  
- 🧽 Clear the canvas with an open palm  
- 🎥 Real-time webcam feed with drawing overlay  
- 💻 Beginner-friendly and fun Computer Vision project  

---

## 🛠️ Tech Stack  
- [Python 3.10+](https://www.python.org/)  
- [OpenCV](https://opencv.org/)  
- [MediaPipe](https://developers.google.com/mediapipe)  
- [NumPy](https://numpy.org/)  

---

## 📦 Installation & Setup  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/air-drawing.git
cd air-drawing

# 2️⃣ Create a virtual environment (recommended)
python -m venv airpaint_env

# 3️⃣ Activate the virtual environment
# For Windows:
.\airpaint_env\Scripts\activate
# For Mac/Linux:
source airpaint_env/bin/activate

# 4️⃣ Install dependencies
pip install opencv-contrib-python mediapipe numpy

# 5️⃣ Run the project
python air_draw.py
