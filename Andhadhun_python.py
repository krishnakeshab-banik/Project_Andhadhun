import cv2
import os
import numpy as np
import pytesseract
import pyttsx3
import time
import speech_recognition as sr
from PIL import Image
from transformers import pipeline
from deepface import DeepFace
from deep_translator import GoogleTranslator  # For translation
import face_recognition  # For face encoding and comparison
import imutils  # For frame resizing
import datetime  # For time and date reporting
from pywinauto import Application  # For automating Phone Link

# Configure Tesseract OCR (adjust path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Constants for Distance Calculation
KNOWN_FACE_WIDTH = 14      # Average human face width in cm
FOCAL_LENGTH_FACE = 500    # Pre-calculated (can be calibrated)
FOCAL_LENGTH_OBJECT = 700  # Pre-calculated (can be calibrated)

# Supported languages mapping (for speech recognition & TTS)
supported_languages = {
    "english": "en-US",
    "hindi": "hi-IN",
    "tamil": "ta-IN"
}
selected_language_code = "en-US"  # default

# Language-specific responses
responses = {
    "en-US": {
        "language_set": "Language set. Starting the system.",
        "listening": "Listening for command.",
        "no_command": "No command detected.",
        "unknown_command": "Could not understand the command.",
        "text_detected": "Text detected: {}",
        "no_text": "No readable text detected.",
        "face_detected": "{} detected at {} centimeters with a mood of {}.",
        "no_obstacle": "No obstacles detected ahead. You can proceed forward.",
        "obstacle_direction": "Obstacles detected. You should go {}.",
        "exit": "Shutting down.",
        "please_say_language": "Please say your preferred language: English, Hindi, or Tamil.",
        "scene": "Scene: {}",
        "object_detected": "I detected: {}.",
        "people_count": "There are {} people.",
        "time_date": "The current time and date is: {}.",
        "call_initiated": "Calling {}."
    },
    "hi-IN": {
        "language_set": "भाषा सेट हो गई है। सिस्टम शुरू हो रहा है।",
        "listening": "आदेश सुन रहा हूँ।",
        "no_command": "कोई आदेश नहीं मिला।",
        "unknown_command": "आदेश समझ में नहीं आया।",
        "text_detected": "पाठ मिला: {}",
        "no_text": "कोई पठनीय पाठ नहीं मिला।",
        "face_detected": "{} को {} सेंटीमीटर दूर, मूड {} के साथ पहचान लिया गया।",
        "no_obstacle": "आगे कोई बाधा नहीं है। आप आगे बढ़ सकते हैं।",
        "obstacle_direction": "बाधाएँ मिली हैं। आपको {} जाना चाहिए।",
        "exit": "बंद किया जा रहा है।",
        "please_say_language": "कृपया अपनी पसंदीदा भाषा कहें: अंग्रेजी, हिंदी, या तमिल।",
        "scene": "दृश्य: {}",
        "object_detected": "मैंने पहचान लिया: {}।",
        "people_count": "यहां {} लोग हैं।",
        "time_date": "अब का समय और तिथि है: {}।",
        "call_initiated": "{} को कॉल किया जा रहा है।"
    },
    "ta-IN": {
        "language_set": "மொழி அமைக்கப்பட்டது. கணினி துவங்குகிறது.",
        "listening": "கட்டளையை கேட்கின்றேன்.",
        "no_command": "எந்த கட்டளையும் பெறவில்லை.",
        "unknown_command": "கட்டளை புரியவில்லை.",
        "text_detected": "உரை கண்டறியப்பட்டது: {}",
        "no_text": "படிக்கக்கூடிய உரை எதுவும் இல்லை.",
        "face_detected": "{}, {} சென்டிமீட்டர்கள் தூரத்தில், உணர்வு {} உடன் கண்டறியப்பட்டது.",
        "no_obstacle": "முன் எந்த தடையும் இல்லை. நீங்கள் முன்னோக்கி செல்லலாம்.",
        "obstacle_direction": "தடைகள் கண்டறியப்பட்டன. நீங்கள் {} செல்ல வேண்டும்.",
        "exit": "முடக்கப்படுகிறது.",
        "please_say_language": "தயவு செய்து உங்கள் விருப்பமான மொழியைச் சொல்லவும்: ஆங்கிலம், ஹிந்தி, அல்லது தமிழ்.",
        "scene": "காட்சி: {}",
        "object_detected": "நான் கண்டுபிடித்தேன்: {}.",
        "people_count": "இங்கே {} பேர் உள்ளனர்.",
        "time_date": "தற்போதைய நேரம் மற்றும் தேதி: {}.",
        "call_initiated": "{}-ஐ அழைக்கிறது."
    }
}

# Global variables for known faces
known_face_encodings = []
known_face_names = []

def load_known_faces(folder_path="known_faces"):
    """
    Loads images from the specified folder, computes face encodings,
    and extracts names from filenames.
    """
    global known_face_encodings, known_face_names
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found. Creating folder.")
        os.makedirs(folder_path)
        print(f"Please add your known faces images (jpg files) into the '{folder_path}' folder and restart the application.")
        exit()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].capitalize()
                known_face_names.append(name)
                print(f"Loaded encoding for {name}")

load_known_faces()

# Initialize cascades for face and object detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize globals for speech, recognition, and timing.
engine = pyttsx3.init()
recognizer = sr.Recognizer()
last_obstacle_time = 0       
last_text = ""               
last_face_time = 0           

# Load MobileNet SSD for object detection.
proto_txt = "MobileNetSSD_deploy.prototxt"
model_file = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_txt, model_file)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize scene captioning pipeline.
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")

# ----- Functions for People Count and Time/Date -----
def report_people_count(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    count = len(face_locations)
    speak_text(responses[selected_language_code]["people_count"].format(count))

def report_time_date():
    now = datetime.datetime.now()
    time_date_str = now.strftime("%I:%M %p, %A, %B %d, %Y")
    speak_text(responses[selected_language_code]["time_date"].format(time_date_str))

# ----- Utility Functions -----
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def set_tts_voice(language_code):
    voices = engine.getProperty('voices')
    target = language_code.split('-')[0]
    print("Available voices:")
    for voice in voices:
        try:
            languages = [lang.decode('utf-8').lower() if isinstance(lang, bytes) else lang.lower() for lang in voice.languages]
        except Exception:
            languages = ["unknown"]
        print("Voice ID:", voice.id, "Languages:", languages, "Name:", voice.name)
    for voice in voices:
        try:
            languages = [lang.decode('utf-8').lower() if isinstance(lang, bytes) else lang.lower() for lang in voice.languages]
            if target in languages[0]:
                engine.setProperty('voice', voice.id)
                print("TTS voice automatically set to:", voice.name)
                return
        except Exception:
            continue
    if language_code == "hi-IN":
        manual_voice_id = "HINDI_VOICE_ID"  # Replace with actual voice ID for Hindi
        print("No automatic Hindi voice found. Manually setting voice to:", manual_voice_id)
        engine.setProperty('voice', manual_voice_id)
    elif language_code == "ta-IN":
        manual_voice_id = "TAMIL_VOICE_ID"  # Replace with actual voice ID for Tamil
        print("No automatic Tamil voice found. Manually setting voice to:", manual_voice_id)
        engine.setProperty('voice', manual_voice_id)
    else:
        print("No matching voice found for", language_code)

def choose_language():
    speak_text(responses["en-US"]["please_say_language"])
    lang_choice = listen_for_command(language="en-US")
    if lang_choice:
        for lang in supported_languages:
            if lang in lang_choice.lower():
                return supported_languages[lang]
    return "en-US"

def listen_for_command(language="en-US"):
    with sr.Microphone() as source:
        print(responses[language]["listening"])
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)
            command = recognizer.recognize_google(audio, language=language).lower()
            print(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            print(responses[language]["unknown_command"])
        except sr.RequestError:
            print("Speech recognition service unavailable.")
        except sr.WaitTimeoutError:
            print("Timeout: No command detected. Please speak louder or try again.")
    return ""

def calculate_distance(focal_length, known_width, perceived_width):
    if perceived_width > 0:
        return (known_width * focal_length) / perceived_width
    return -1

def describe_scene(frame):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = captioner(pil_image)
        if result and isinstance(result, list) and "generated_text" in result[0]:
            description = result[0]["generated_text"]
            if selected_language_code == "hi-IN":
                description = GoogleTranslator(source='en', target='hi').translate(description)
            elif selected_language_code == "ta-IN":
                description = GoogleTranslator(source='en', target='ta').translate(description)
            return description
    except Exception as e:
        print(f"Error in scene description: {e}")
    return "No description."

def detect_text(frame):
    global last_text
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
    text = pytesseract.image_to_string(gray, config='--psm 6').strip()
    if text and text != last_text:
        speak_text(responses[selected_language_code]["text_detected"].format(text))
        last_text = text
    elif not text:
        speak_text(responses[selected_language_code]["no_text"])

def navigate_myself(frame):
    global last_obstacle_time
    current_time = time.time()
    if current_time - last_obstacle_time < 5:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    obstacles = object_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    frame_width = frame.shape[1]
    if len(obstacles) == 0:
        speak_text(responses[selected_language_code]["no_obstacle"])
        last_obstacle_time = current_time
        return
    zones = {"left": {"count": 0, "area": 0},
             "center": {"count": 0, "area": 0},
             "right": {"count": 0, "area": 0}}
    for (x, y, w, h) in obstacles:
        cx = x + w / 2
        area = w * h
        if cx < frame_width / 3:
            zones["left"]["count"] += 1
            zones["left"]["area"] += area
        elif cx < 2 * frame_width / 3:
            zones["center"]["count"] += 1
            zones["center"]["area"] += area
        else:
            zones["right"]["count"] += 1
            zones["right"]["area"] += area
    best_zone = None
    best_count = float('inf')
    best_area = float('inf')
    for zone, stats in zones.items():
        if stats["count"] < best_count or (stats["count"] == best_count and stats["area"] < best_area):
            best_zone = zone
            best_count = stats["count"]
            best_area = stats["area"]
    if selected_language_code == "en-US":
        direction = best_zone
    elif selected_language_code == "hi-IN":
        direction = "बाएं" if best_zone == "left" else ("दाएं" if best_zone == "right" else "सीधे")
    elif selected_language_code == "ta-IN":
        direction = "இடப்புறம்" if best_zone == "left" else ("வலப்புறம்" if best_zone == "right" else "நேராக")
    speak_text(responses[selected_language_code]["obstacle_direction"].format(direction))
    last_obstacle_time = current_time

def detect_mood(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        mood = analysis[0]['dominant_emotion']
        return mood
    except Exception as e:
        print(f"Error detecting mood: {e}")
        return "Unknown"

def identify_object(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    found_objects = set()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label.lower() != "person":
                found_objects.add(label)
    if found_objects:
        speak_text(responses[selected_language_code]["object_detected"].format(", ".join(found_objects)))
    else:
        speak_text(responses[selected_language_code]["no_obstacle"])

def identify_people(frame):
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    messages = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_width_pixels = right - left
        face_distance = calculate_distance(FOCAL_LENGTH_FACE, KNOWN_FACE_WIDTH, face_width_pixels)
        mood = detect_mood(frame)
        messages.append(responses[selected_language_code]["face_detected"].format(name, int(face_distance), mood))
    if messages:
        speak_text(" ".join(messages))
    else:
        speak_text("No faces detected.")

# Helper function: Get the first ListItem control.
def get_first_contact_item(main_window):
    list_items = main_window.descendants(control_type="ListItem")
    if list_items:
        print("Found the following list items:")
        for item in list_items:
            try:
                item_text = item.window_text().strip()
                print(f"List item: '{item_text}'")
            except Exception as e:
                print("Error reading list item:", e)
        return list_items[0]  # Return the first item directly
    return None

# Updated call_contact function that selects the first contact after search and places the call.
def call_contact(contact_name):
    """Automates the Phone Link app to call a contact using the 'Calls' tab."""
    from pywinauto import Application
    try:
        app = Application(backend="uia").connect(title_re=".*Phone Link.*")
        print("Connected to Phone Link.")
    except Exception as e:
        print("Phone Link not running. Please open it manually.", e)
        return

    main_window = app.window(title_re=".*Phone Link.*")
    main_window.wait('visible', timeout=30)
    print("Phone Link main window is ready.")
    
    # Debug: Print descendant controls for inspection
    try:
        for ctrl in main_window.descendants():
            try:
                txt = ctrl.window_text().strip()
                print(f"Control: '{txt}', Type: {ctrl.friendly_class_name()}")
            except Exception:
                continue
    except Exception as e:
        print("Error printing control identifiers:", e)
    
    # Attempt to locate the Calls tab by searching for any control with text containing "call" or "dial"
    calls_tab = None
    for ctrl in main_window.descendants():
        try:
            txt = ctrl.window_text().strip()
            if txt and ("call" in txt.lower() or "dial" in txt.lower()):
                calls_tab = ctrl.wrapper_object()
                print(f"Found potential Calls tab: '{txt}'")
                break
        except Exception:
            continue
    if not calls_tab:
        print("Could not find Calls tab. Assuming the app is already in the Calls view.")
    else:
        try:
            calls_tab.click_input()
            print("Calls tab clicked.")
            time.sleep(3)
        except Exception as e:
            print("Error clicking Calls tab:", e)
            return
    
    # Locate the search box and input the contact name.
    try:
        search_box = main_window.child_window(control_type="Edit", found_index=0).wrapper_object()
        search_box.set_edit_text(contact_name)
        print(f"Searching for {contact_name}...")
    except Exception as e:
        print("Error locating search box:", e)
        return
    time.sleep(3)
    
    # Select the first contact from the search results.
    contact_item = get_first_contact_item(main_window)
    if not contact_item:
        print("Error selecting contact: No list item found.")
        return

    try:
        contact_item.click_input()
        print(f"Selected first contact in list for {contact_name}")
    except Exception as e:
        print("Error clicking on contact item:", e)
        return
    time.sleep(2)
    
    # Click the "Call" button to initiate the call.
    try:
        call_button = main_window.child_window(title_re=".*Call.*", control_type="Button").wrapper_object()
        call_button.click_input()
        print("Call initiated.")
        speak_text(responses[selected_language_code]["call_initiated"].format(contact_name))
    except Exception as e:
        print("Error clicking Call button:", e)

def report_people_count(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    count = len(face_locations)
    speak_text(responses[selected_language_code]["people_count"].format(count))

def report_time_date():
    now = datetime.datetime.now()
    time_date_str = now.strftime("%I:%M %p, %A, %B %d, %Y")
    speak_text(responses[selected_language_code]["time_date"].format(time_date_str))

def recognize_and_navigate():
    global last_face_time
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    
    while True:
        try:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                continue

            frame = imutils.resize(frame, width=600)
            rgb_frame = frame[:, :, ::-1]
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            current_time = time.time()
            
            # Automatic face recognition every 5 seconds.
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_width_pixels = right - left
                face_distance = calculate_distance(FOCAL_LENGTH_FACE, KNOWN_FACE_WIDTH, face_width_pixels)
                mood = detect_mood(frame)
                if current_time - last_face_time > 5:
                    scene_desc = describe_scene(frame)
                    message = responses[selected_language_code]["face_detected"].format(name, int(face_distance), mood)
                    message += " " + responses[selected_language_code]["scene"].format(scene_desc)
                    speak_text(message)
                    last_face_time = current_time
            
            cv2.putText(frame, "Commands: 'Read Text' | 'Describe' | 'Navigate' | 'Identify Object' | 'Identify People' | 'How Many People' | 'Time & Date' | 'Call [Name]' | 'Exit'", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            
            command = listen_for_command(language=selected_language_code)
            if selected_language_code == "hi-IN":
                if "पाठ पढ़ें" in command:
                    detect_text(frame)
                elif "मेरे सामने क्या है" in command:
                    desc = describe_scene(frame)
                    speak_text(responses[selected_language_code]["scene"].format(desc))
                elif "मुझे रास्ता दिखाओ" in command:
                    navigate_myself(frame)
                elif "वस्तु पहचानें" in command:
                    identify_object(frame)
                elif "लोग पहचानें" in command:
                    identify_people(frame)
                elif "कितने लोग" in command:
                    report_people_count(frame)
                elif "समय" in command and "तिथि" in command:
                    report_time_date()
                elif "कॉल" in command:
                    parts = command.split("कॉल", 1)
                    if len(parts) > 1:
                        contact = parts[1].strip().capitalize()
                        call_contact(contact)
                    else:
                        speak_text("No contact name provided.")
                elif "बंद करें" in command or "बंद" in command:
                    speak_text(responses[selected_language_code]["exit"])
                    break
            elif selected_language_code == "ta-IN":
                if "உரை வாசி" in command:
                    detect_text(frame)
                elif "காட்சி விவரிக்க" in command:
                    desc = describe_scene(frame)
                    speak_text(responses[selected_language_code]["scene"].format(desc))
                elif "வழிசெய்" in command:
                    navigate_myself(frame)
                elif "பொருளை அடையாளம் காணுங்கள்" in command:
                    identify_object(frame)
                elif "மக்களை அடையாளம் காணுங்கள்" in command:
                    identify_people(frame)
                elif "எத்தனை பேர்" in command:
                    report_people_count(frame)
                elif "நேரமும்" in command and "தேதி" in command:
                    report_time_date()
                elif "கால்" in command:
                    parts = command.split("கால்", 1)
                    if len(parts) > 1:
                        contact = parts[1].strip().capitalize()
                        call_contact(contact)
                    else:
                        speak_text("No contact name provided.")
                elif "முடிக்க" in command:
                    speak_text(responses[selected_language_code]["exit"])
                    break
            else:
                if "read text" in command:
                    detect_text(frame)
                elif "describe" in command:
                    desc = describe_scene(frame)
                    speak_text(responses[selected_language_code]["scene"].format(desc))
                elif "navigate" in command:
                    navigate_myself(frame)
                elif "identify object" in command:
                    identify_object(frame)
                elif "identify people" in command:
                    identify_people(frame)
                elif "how many people" in command:
                    report_people_count(frame)
                elif "time" in command and "date" in command:
                    report_time_date()
                elif "call" in command:
                    parts = command.split("call", 1)
                    if len(parts) > 1:
                        contact = parts[1].strip().capitalize()
                        call_contact(contact)
                    else:
                        speak_text("No contact name provided.")
                elif "exit" in command or "quit" in command:
                    speak_text(responses[selected_language_code]["exit"])
                    break
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    selected_language_code = choose_language()
    set_tts_voice(selected_language_code)
    speak_text(responses[selected_language_code]["language_set"])
    recognize_and_navigate()
