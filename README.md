# 🧠 Hand Tremor Detection & Severity Analyzer

A real-time medical screening prototype that detects hand tremors using OpenCV, MediaPipe, and Streamlit,
and generates a downloadable PDF tremor report with patient details and graphs.

⚠️ Educational & screening use only. Not a diagnostic medical device.

## Features
- Real-time tremor detection via webcam (local run)
- MediaPipe hand landmark tracking
- Live tremor graph
- Severity classification (Normal / Mild / Severe)


## Tech Stack
Python 3.10, OpenCV, MediaPipe, Streamlit, NumPy, Matplotlib, ReportLab

## Run Locally
```bash
python3.10 -m venv tremor_env
source tremor_env/bin/activate
pip install -r requirements.txt
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
streamlit run tremor.py
```

