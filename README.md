# License-Plate-Detection-System
This repository contains the code for an automated vehicle and license plate detection system developed to assist in traffic management, law enforcement, and parking facility monitoring by utilizing advanced machine learning models and computer vision techniques. The system processes video feeds to identify vehicles and their license plates in real-time, supporting various surveillance operations.

## Features
1. Real-time Vehicle Detection: Detect multiple vehicle types across different environments using the YOLO object detection framework.
2. License Plate Recognition: Identify and extract license plate information from detected vehicles for further processing.
3. Vehicle Tracking: Maintain vehicle identity across frames using SORT tracking technology.
4. High Accuracy and Efficiency: Designed to achieve high detection accuracy and processing speed for real-time applications.
5. Scalable Solution: Ready to integrate into existing surveillance infrastructures without requiring significant additional resources.

## Prerequisites
Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- OpenCV library
- Ultralytics YOLO models

Optionally, other dependencies as specified in the requirements.txt file.

## Installation
To install the necessary libraries, follow these steps:
`pip install -r requirements.txt`

### Usage
To use this detection system, follow these steps:

Clone the repository:
```git clone https://github.com/yourusername/vehicle-license-detection.git```

Navigate to the project directory:
```cd vehicle-license-detection```

Activate the virtual environment:
```source venv/bin/activate```  # On Unix or MacOS
```.\venv\Scripts\activate```    # On Windows

Run the main script:
```python main.py```

The system will start processing the video specified in the script and display the results in real-time. Adjust the path to your video file within main.py as needed.

Configuration
Modify the main.py to point to different video sources or adjust the detection and tracking parameters according to your specific requirements.

Contributing
Contributions to enhance the functionality of this system are welcome. Please adhere to the following steps:

Fork the repository.
Create a new branch (git checkout -b feature/AmazingFeature).
Make your changes.
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
