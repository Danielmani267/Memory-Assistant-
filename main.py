from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import uvicorn
from deepface import DeepFace
import os
import json
from datetime import datetime
import threading
import time
from gtts import gTTS
import tempfile
from twilio.rest import Client

app = FastAPI()

KNOWN_FACES_DIR = "known_faces"
MEMORY_DB_FILE  = "memory_db.json"

# Twilio credentials
TWILIO_SID    = "ACfabbb02e21146dd665783eb6134097a5"
TWILIO_TOKEN  = "06fd2130727f41b4e8b45ecfe6864f90"
TWILIO_NUMBER = "whatsapp:+14155238886"
FAMILY_NUMBER = "whatsapp:+919036160299"

last_spoken      = ""
last_spoken_time = 0
last_alert_time  = 0

def send_whatsapp_alert(message):
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(
            body = message,
            from_= TWILIO_NUMBER,
            to   = FAMILY_NUMBER
        )
        print(f"✅ WhatsApp alert sent!")
    except Exception as e:
        print(f"❌ WhatsApp error: {e}")

def load_memory_db():
    if os.path.exists(MEMORY_DB_FILE):
        with open(MEMORY_DB_FILE, "r",
                  encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory_db(db):
    with open(MEMORY_DB_FILE, "w",
              encoding="utf-8") as f:
        json.dump(db, f, indent=2,
                  ensure_ascii=False)

def recognize_face(image_path):
    try:
        result = DeepFace.find(
            img_path          = image_path,
            db_path           = KNOWN_FACES_DIR,
            enforce_detection = False,
            model_name        = "VGG-Face",
            threshold         = 0.4
        )
        if len(result) > 0 and len(result[0]) > 0:
            distance = result[0].iloc[0]["distance"]
            print(f"📊 Distance: {distance}")
            if distance < 0.4:
                matched_path = result[0].iloc[0]["identity"]
                person_name  = matched_path.split(os.sep)[-2]
                print(f"✅ Recognized: {person_name}")
                return person_name
            else:
                print(f"❌ Unknown! Distance: {distance}")
                return None
        return None
    except Exception as e:
        print(f"Recognition error: {e}")
        return None

def detect_emotion(image_path):
    try:
        result = DeepFace.analyze(
            img_path          = image_path,
            actions           = ['emotion'],
            enforce_detection = False
        )
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Emotion error: {e}")
        return None
def emotion_message(emotion, language):
    messages = {
        "en": {
            "happy"   : "They look very happy today!",
            "sad"     : "They seem a little sad. Be kind!",
            "angry"   : "They look upset today.",
            "surprise": "They look surprised!",
            "fear"    : "They look nervous.",
            "neutral" : "They look calm today."
        },
        "kn": {
            "happy"   : "ಅವರು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿ ಕಾಣುತ್ತಿದ್ದಾರೆ!",
            "sad"     : "ಅವರು ಸ್ವಲ್ಪ ದುಃಖಿತರಾಗಿ ಕಾಣುತ್ತಿದ್ದಾರೆ.",
            "angry"   : "ಅವರು ಇಂದು ಕೋಪದಲ್ಲಿ ಕಾಣುತ್ತಿದ್ದಾರೆ.",
            "surprise": "ಅವರು ಆಶ್ಚರ್ಯಚಕಿತರಾಗಿ ಕಾಣುತ್ತಿದ್ದಾರೆ!",
            "fear"    : "ಅವರು ಆತಂಕದಲ್ಲಿ ಕಾಣುತ್ತಿದ್ದಾರೆ.",
            "neutral" : "ಅವರು ಇಂದು ಶಾಂತವಾಗಿ ಕಾಣುತ್ತಿದ್ದಾರೆ."
        },
        "hi": {
            "happy"   : "वो आज बहुत खुश लग रहे हैं!",
            "sad"     : "वो थोड़े उदास लग रहे हैं।",
            "angry"   : "वो आज नाराज़ लग रहे हैं।",
            "surprise": "वो हैरान लग रहे हैं!",
            "fear"    : "वो घबराए हुए लग रहे हैं।",
            "neutral" : "वो आज शांत लग रहे हैं।"
        },
        "ta": {
            "happy"   : "அவர் இன்று மிகவும் மகிழ்ச்சியாக இருக்கிறார்!",
            "sad"     : "அவர் கொஞ்சம் சோகமாக தெரிகிறார்.",
            "angry"   : "அவர் இன்று கோபமாக தெரிகிறார்.",
            "surprise": "அவர் ஆச்சரியமாக தெரிகிறார்!",
            "fear"    : "அவர் பயமாக தெரிகிறார்.",
            "neutral" : "அவர் இன்று அமைதியாக தெரிகிறார்."
        },
        "te": {
            "happy"   : "వారు ఈరోజు చాలా సంతోషంగా కనిపిస్తున్నారు!",
            "sad"     : "వారు కొంచెం దుఃఖంగా కనిపిస్తున్నారు.",
            "angry"   : "వారు ఈరోజు కోపంగా కనిపిస్తున్నారు.",
            "surprise": "వారు ఆశ్చర్యంగా కనిపిస్తున్నారు!",
            "fear"    : "వారు భయంగా కనిపిస్తున్నారు.",
            "neutral" : "వారు ఈరోజు శాంతంగా కనిపిస్తున్నారు."
        }
    }
    lang_messages = messages.get(language, messages["en"])
    return lang_messages.get(emotion, "")

def greeting(language):
    greetings = {
        "en": "Hi Ganesh! This is",
        "kn": "ಹಾಯ್ ಗಣೇಶ್! ಇವರು",
        "hi": "नमस्ते गणेश! ये हैं",
        "ta": "வணக்கம் கணேஷ்! இவர்",
        "te": "హలో గణేష్! ఇతను"
    }
    return greetings.get(language, greetings["en"])

def relationship_text(relationship, language):
    relationships = {
        "kn": {
            "best friend": "ನಿಮ್ಮ ಆತ್ಮೀಯ ಗೆಳೆಯ",
            "son"        : "ನಿಮ್ಮ ಮಗ",
            "doctor"     : "ನಿಮ್ಮ ವೈದ್ಯರು"
        },
        "hi": {
            "best friend": "आपके प्रिय मित्र",
            "son"        : "आपके बेटे",
            "doctor"     : "आपके डॉक्टर"
        },
        "ta": {
            "best friend": "உங்கள் நெருங்கிய நண்பர்",
            "son"        : "உங்கள் மகன்",
            "doctor"     : "உங்கள் மருத்துவர்"
        },
        "te": {
            "best friend": "మీ ప్రియమైన స్నేహితుడు",
            "son"        : "మీ కొడుకు",
            "doctor"     : "మీ వైద్యుడు"
        }
    }
    lang_rel = relationships.get(language, {})
    return lang_rel.get(relationship, relationship)

def log_activity(person_name, emotion=None):
    log_file = "activity_log.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r",
                  encoding="utf-8") as f:
            logs = json.load(f)
    logs.append({
        "person"   : person_name,
        "emotion"  : emotion,
        "timestamp": datetime.now().strftime(
                     "%Y-%m-%d %H:%M:%S")
    })
    with open(log_file, "w",
              encoding="utf-8") as f:
        json.dump(logs, f, indent=2,
                  ensure_ascii=False)

@app.post("/recognize")
async def recognize(request: Request):
    global last_spoken
    global last_spoken_time
    global last_alert_time

    image_bytes = await request.body()
    image_path  = "temp_capture.jpg"

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    person_name = recognize_face(image_path)
    emotion     = detect_emotion(image_path)
    print(f"😊 Emotion: {emotion}")

    if person_name:
        memory_db     = load_memory_db()
        person_memory = memory_db.get(person_name, {})
        name          = person_memory.get("name", person_name)
        relationship  = person_memory.get("relationship", "person")
        language      = person_memory.get("language", "en")
        notes_key     = f"notes_{language}"
        notes         = person_memory.get(
                        notes_key,
                        person_memory.get("notes_en", ""))

        greet        = greeting(language)
        relation_txt = relationship_text(relationship, language)
        emotion_msg  = emotion_message(
                       emotion, language) if emotion else ""

        response_text = (
            f"{greet} {name} {relation_txt}. "
            f"{notes} {emotion_msg}"
        )

        current_time = time.time()
        if (last_spoken != person_name or
                (current_time - last_spoken_time) > 30):
            last_spoken      = person_name
            last_spoken_time = current_time
            print(f"🔊 Response ({language}): {response_text}")

        log_activity(person_name, emotion)

        return JSONResponse({
            "status"       : "recognized",
            "person"       : person_name,
            "emotion"      : emotion,
            "language"     : language,
            "response_text": response_text,
            "relationship" : relationship,
            "notes"        : notes
        })

    else:
        last_spoken      = ""
        last_spoken_time = 0

        unknown_texts = {
            "en": "Unknown person detected! Please be careful!",
            "kn": "ಅಪರಿಚಿತ ವ್ಯಕ್ತಿ ಪತ್ತೆಯಾಗಿದ್ದಾರೆ! ಎಚ್ಚರಿಕೆಯಿಂದಿರಿ!",
            "hi": "अनजान व्यक्ति मिला! सावधान रहें!",
            "ta": "அறியாத நபர் கண்டறியப்பட்டார்! கவனமாக இருங்கள்!",
            "te": "తెలియని వ్యక్తి గుర్తించబడ్డారు! జాగ్రత్తగా ఉండండి!"
        }
        unknown_msg = unknown_texts.get("kn")

        current_time = time.time()
        if (current_time - last_alert_time) > 60:
            now = datetime.now().strftime(
                  "%Y-%m-%d %H:%M:%S")
            alert_msg = (
                f"🚨 *ALERT - Memory Assistant*\n\n"
                f"Unknown person detected near Ganesh!\n\n"
                f"⏰ Time: {now}\n\n"
                f"Please check immediately!\n\n"
                f"- Memory Assistant System"
            )
            threading.Thread(
                target=send_whatsapp_alert,
                args=(alert_msg,)
            ).start()
            last_alert_time = current_time

        return JSONResponse({
            "status"       : "unknown",
            "person"       : None,
            "emotion"      : emotion,
            "response_text": unknown_msg,
            "relationship" : None,
            "notes"        : None
        })

@app.post("/register")
async def register_person(
    name: str,
    relationship: str,
    notes: str,
    language: str = "en",
    file: UploadFile = File(...)
):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    image_bytes = await file.read()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path  = os.path.join(person_dir, f"{timestamp}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    memory_db = load_memory_db()
    memory_db[name] = {
        "name"        : name,
        "relationship": relationship,
        "notes_en"    : notes,
        "language"    : language,
        "photos"      : memory_db.get(
                        name, {}).get(
                        "photos", []) + [image_path]
    }
    save_memory_db(memory_db)

    return JSONResponse({
        "status" : "success",
        "message": f"{name} registered successfully!"
    })

@app.get("/people")
async def get_people():
    memory_db = load_memory_db()
    return JSONResponse(memory_db)

@app.get("/activity")
async def get_activity():
    log_file = "activity_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r",
                  encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    return JSONResponse([])

if __name__ == "__main__":
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
