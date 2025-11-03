# 🎓 FocusMonitor – AI-Powered Student Attention Tracker  

**FocusMonitor** is an intelligent real-time attention tracking system built using **Flask**, **OpenCV**, and **Deep Learning**.  
It monitors students’ eye movements and head orientation through a webcam feed to determine their level of focus during online or offline study sessions.  

---

## 🚀 Features
- 🎥 **Live Camera Feed:** Detects student presence and monitors focus in real time.  
- 📈 **Focus Meter:** Displays live attention score dynamically.  
- 🔔 **Smart Alerts:** Shows “⚠ Stay Alert!” pop-ups when focus drops.  
- 🌗 **Dark/Light Mode:** Modern toggle for comfort and aesthetics.  
- 📊 **Focus Graph:** Visualizes focus trends over the last 30 seconds.  
- 🎵 **Sound Feedback:** Subtle alert sound when drowsiness or distraction is detected.  

---

## 🧠 How It Works
1. The webcam captures live video frames.  
2. Facial landmarks are detected using a pre-trained model (`shape_predictor_68_face_landmarks.dat`).  
3. Eye Aspect Ratio (EAR) and head position are analyzed to estimate focus level.  
4. If the focus score drops below a certain threshold, an alert is triggered visually and audibly.  

---

## 💻 Tech Stack
- **Frontend:** HTML, CSS, JavaScript (Chart.js for graphs)  
- **Backend:** Python (Flask Framework)  
- **Libraries:** OpenCV, dlib, imutils, NumPy  
- **Model:** 68-point facial landmark detector  

---


🧩 Applications

🧑‍🏫 Online classroom attention monitoring

🎧 Study or focus companion app

💼 Corporate training attention analysis

🚗 Extended to driver drowsiness detection systems


📚 References

Adrian Rosebrock – PyImageSearch Blog: Facial Landmark Detection with dlib

Research Paper: “Eye Aspect Ratio for Fatigue Detection using Facial Landmarks”

Chart.js Documentation

Flask Official Docs

OpenCV Official Documentation


🔮 Future Improvements

🧠 Deep Learning Integration: Replace traditional landmark detection with a CNN-based facial emotion and focus recognition model for higher accuracy.

📷 Multi-Face Detection: Support multiple students simultaneously for classroom-level monitoring.

🎵 Smart Alert System: Personalized sound or voice feedback instead of generic alerts.

📊 Data Analytics Dashboard: Track long-term focus trends, generate weekly or monthly reports.

☁️ Cloud Integration: Save focus history and sync data across sessions using Firebase or AWS.

📱 Mobile Compatibility: Extend the app to Android/iOS using Flask API + React Native.

💬 AI Chat Assistant: Provide study reminders and focus tips based on user patterns.

🔐 Privacy Controls: Add local data encryption and user consent settings.



📜 License

This project is open-source and available under the MIT License.

Author:
Atul Chand Kapri
📫 Developer | Problem Solver | Web + AI Enthusiast
🔗 GitHub – AtulKapri05

