# ANDHADHUN - AI-Powered Smart Glasses for the Visually Impaired

## Introduction
ANDHADHUN is an AI-powered smart glasses project designed to assist visually impaired individuals in navigating their surroundings and interacting with people and objects independently. The glasses integrate facial recognition, obstacle detection, and a real-time AI voice assistant, providing real-time voice feedback to enhance safety and social interaction.

## Features
- **Facial Recognition**: Identifies known individuals using a predefined database.
- **Obstacle Detection**: Alerts users about obstacles and provides directional guidance.
- **AI Voice Assistant**: Responds to user queries and provides real-time scene descriptions.
- **Multilingual Support**: Supports English, Hindi, and Tamil.
- **Text Detection**: Reads and vocalizes text from the environment using OCR.
- **Scene Description**: Uses AI to describe the surroundings in real time.
- **Voice-Controlled Calls**: Automates calling contacts via a connected phone.
- **Time & Date Reporting**: Provides real-time updates on time and date.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy pytesseract pyttsx3 speechrecognition transformers deepface deep-translator face-recognition imutils pywinauto torch torchvision torchaudio
```

You will also need:
- **Tesseract OCR**: Install from [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Pretrained Face Data**: Place known faces in a `known_faces` folder.
- **MobileNet SSD Model**: Ensure `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` are in the project directory.
- **YOLO Model**: Download the YOLO model weights (`yolov3.weights`) and config files (`yolov3.cfg`) from [YOLO Website](https://pjreddie.com/darknet/yolo/) and place them in the `yolo` directory.
- **YOLO Class Labels**: Download `coco.names` from the same source and place it in the `yolo` directory.
- **Haarcascade Files**: Ensure `haarcascade_frontalface_default.xml` and `haarcascade_fullbody.xml` are present in the project directory for face and object detection.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/andhadhun_python.git
   cd andhadhun_python
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Tesseract OCR path in `Andhadhun_python.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
4. Download YOLO model and place files in the `yolo` directory.

## Usage
Run the script:
```bash
python Andhadhun_python.py
```
After running, follow the voice prompts to:
- Recognize faces
- Detect and avoid obstacles
- Read text from the environment
- Identify objects and describe scenes
- Make calls using voice commands
- Change language settings

### Example Commands
- "Read Text"
- "Describe the scene"
- "Navigate"
- "Identify object"
- "How many people?"
- "Call [Name]"
- "Time and date"
- "Exit"

## Contributions
### Team HUNT-X
- **Shaurya Kesarwani** - Technical
- **Krishna Keshab Banik** - Technical Development
- **Aaryan Sarat** - Hardware Development
- **Dhriti Kothari Jain** - Presentation and Research
- **Sanjay Kumar Gupta** - Presentation and Research

Feel free to contribute by submitting pull requests or reporting issues.

## License
This project is licensed under the Proprietary License. Usage requires explicit permission.

## Contact
For queries and support, contact **Krishna Keshab Banik** at [krishna.keshab.banik@gmail.com](mailto:krishna.keshab.banik@gmail.com).
