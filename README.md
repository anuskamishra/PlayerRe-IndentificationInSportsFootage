# 🧠 Player Re-Identification in a Single Feed 🎥

This project performs real-time **player detection**, **tracking**, and **re-identification** in a 15-second video using **YOLOv8** and **Deep SORT**.

Even when players go out of frame and return, the system keeps the **same Player ID** across frames.

---

## ✅ Objective

- Detect all players in the given input video (`15sec_input_720p.mp4`)
- Assign a **unique ID** to each player
- Maintain ID consistency even when players **disappear and re-enter** the frame

---

## 📦 Requirements

- Python 3.8+
- pip

Install dependencies:

```bash
pip install ultralytics opencv-python deep_sort_realtime
