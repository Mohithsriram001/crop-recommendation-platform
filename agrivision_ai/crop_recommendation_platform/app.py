from pathlib import Path
import uuid

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

try:
    from PIL import Image
except ImportError:
    Image = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_crop_model.joblib"

app = Flask(__name__)
app.secret_key = "agriculture_ai_secret_key"
app.config["UPLOAD_FOLDER"] = BASE_DIR / "static" / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

model = joblib.load(MODEL_PATH)


APP_NAME = "AgriVision AI"
APP_TAGLINE = "Smart Farming Assistant"

SOIL_OPTIONS = [
    "Black Soil",
    "Red Soil",
    "Alluvial Soil",
    "Clay Soil",
    "Sandy Soil",
    "Loamy Soil",
    "Sandy Loam Soil",
    "Clay Loam Soil",
    "Laterite Soil",
]

SEASON_OPTIONS = [
    "Kharif",
    "Rabi",
    "Zaid / Summer",
    "Winter",
    "Rainy",
    "All Season",
]

IRRIGATION_OPTIONS = [
    "Rainfed",
    "Borewell",
    "Canal Irrigation",
    "Drip Irrigation",
    "Sprinkler Irrigation",
    "Flood Irrigation",
    "Manual Irrigation",
    "Tank / Pond Irrigation",
]

DEMO_USER = {
    "name": "Demo Farmer",
    "email": "farmer@example.com",
    "phone": "+91 98765 43210",
    "location": "Coimbatore, Tamil Nadu",
    "land_area": "3.2 acres",
    "experience": "6 years",
    "focus_crop": "Rice and Banana",
}

NAV_ITEMS = [
    {"endpoint": "home", "label": "Home", "active_endpoints": ["home"]},
    {"endpoint": "dashboard", "label": "Dashboard", "active_endpoints": ["dashboard"]},
    {
        "endpoint": "crop_intelligence",
        "label": "Crop Intelligence",
        "active_endpoints": ["crop_intelligence", "predict", "crop", "farmer"],
    },
    {
        "endpoint": "plant_health",
        "label": "Plant Health",
        "active_endpoints": ["plant_health", "disease"],
    },
    {"endpoint": "weather", "label": "Weather Advisory", "active_endpoints": ["weather"]},
    {"endpoint": "chatbot", "label": "AI Chatbot", "active_endpoints": ["chatbot"]},
    {
        "endpoint": "profile",
        "label": "Profile",
        "active_endpoints": ["profile", "edit_profile", "login", "register", "logout"],
    },
    {"endpoint": "about", "label": "About", "active_endpoints": ["about"]},
]

CROP_DETAILS = {
    "apple": {
        "name": "Apple",
        "water": "Medium",
        "season": "Winter",
        "soil": "Loamy soil",
        "duration": "4 to 5 years for fruiting",
        "summary": "Suitable for cooler conditions with balanced soil moisture and a stable nutrient profile.",
    },
    "banana": {
        "name": "Banana",
        "water": "High",
        "season": "All season",
        "soil": "Rich loamy soil",
        "duration": "9 to 12 months",
        "summary": "Performs best with steady moisture, warm temperatures, and regular nutrient support.",
    },
    "blackgram": {
        "name": "Black Gram",
        "water": "Medium",
        "season": "Kharif / Rabi",
        "soil": "Black soil / Loamy soil",
        "duration": "70 to 90 days",
        "summary": "A strong pulse option for balanced soils and moderate irrigation planning.",
    },
    "chickpea": {
        "name": "Chickpea",
        "water": "Low to Medium",
        "season": "Rabi",
        "soil": "Sandy loam soil",
        "duration": "95 to 110 days",
        "summary": "Well suited to drier winter windows and soils with good drainage.",
    },
    "coconut": {
        "name": "Coconut",
        "water": "High",
        "season": "All season",
        "soil": "Sandy loam / Coastal soil",
        "duration": "5 to 6 years for fruiting",
        "summary": "Thrives where moisture availability stays consistent and roots can expand freely.",
    },
    "coffee": {
        "name": "Coffee",
        "water": "Medium to High",
        "season": "Winter / Rainy",
        "soil": "Well-drained loamy soil",
        "duration": "3 to 4 years for yield",
        "summary": "Prefers moderated temperatures, steady humidity, and careful shade-based crop management.",
    },
    "cotton": {
        "name": "Cotton",
        "water": "Medium",
        "season": "Kharif",
        "soil": "Black soil",
        "duration": "150 to 180 days",
        "summary": "A practical match for warmer growing windows with structured irrigation and field monitoring.",
    },
    "grapes": {
        "name": "Grapes",
        "water": "Medium",
        "season": "Rabi / Summer",
        "soil": "Well-drained sandy loam soil",
        "duration": "120 to 150 days",
        "summary": "Benefits from drainage control, disease scouting, and steady canopy management.",
    },
    "jute": {
        "name": "Jute",
        "water": "High",
        "season": "Kharif",
        "soil": "Alluvial soil",
        "duration": "120 to 150 days",
        "summary": "Favors moisture-rich conditions and fertile soil with good monsoon support.",
    },
    "kidneybeans": {
        "name": "Kidney Beans",
        "water": "Medium",
        "season": "Rabi",
        "soil": "Loamy soil",
        "duration": "90 to 120 days",
        "summary": "A balanced legume option when drainage, moderate watering, and nutrient discipline are in place.",
    },
    "lentil": {
        "name": "Lentil",
        "water": "Low",
        "season": "Rabi",
        "soil": "Loamy soil",
        "duration": "100 to 120 days",
        "summary": "Supports efficient winter-season cropping where moisture stress is kept low.",
    },
    "maize": {
        "name": "Maize",
        "water": "Medium",
        "season": "Kharif / Rabi",
        "soil": "Well-drained loamy soil",
        "duration": "90 to 120 days",
        "summary": "A versatile choice for moderate rainfall zones with balanced NPK conditions.",
    },
    "mango": {
        "name": "Mango",
        "water": "Medium",
        "season": "Summer",
        "soil": "Well-drained loamy soil",
        "duration": "3 to 5 years for fruiting",
        "summary": "Matches warm-field conditions where drainage and orchard planning are carefully managed.",
    },
    "mothbeans": {
        "name": "Moth Beans",
        "water": "Low",
        "season": "Kharif",
        "soil": "Sandy soil",
        "duration": "75 to 90 days",
        "summary": "Useful in drier regions where water-saving crop planning is important.",
    },
    "mungbean": {
        "name": "Mung Bean",
        "water": "Low to Medium",
        "season": "Kharif / Summer",
        "soil": "Loamy soil",
        "duration": "60 to 75 days",
        "summary": "Fast-growing and suitable for short-duration crop cycles with moderate nutrient support.",
    },
    "muskmelon": {
        "name": "Muskmelon",
        "water": "Medium",
        "season": "Summer",
        "soil": "Sandy loam soil",
        "duration": "70 to 90 days",
        "summary": "Performs well in warm weather with planned irrigation and healthy root-zone aeration.",
    },
    "orange": {
        "name": "Orange",
        "water": "Medium",
        "season": "Winter / Summer",
        "soil": "Well-drained loamy soil",
        "duration": "6 to 8 months",
        "summary": "Needs balanced watering and strong drainage to support fruit quality and root health.",
    },
    "papaya": {
        "name": "Papaya",
        "water": "Medium to High",
        "season": "All season",
        "soil": "Well-drained sandy loam soil",
        "duration": "8 to 10 months",
        "summary": "Adapts well when warmth, drainage, and regular nutrient replenishment stay consistent.",
    },
    "pigeonpeas": {
        "name": "Pigeon Peas",
        "water": "Low to Medium",
        "season": "Kharif",
        "soil": "Black soil / Loamy soil",
        "duration": "120 to 180 days",
        "summary": "A resilient pulse recommendation for moderate rainfall zones and long-duration planning.",
    },
    "pomegranate": {
        "name": "Pomegranate",
        "water": "Low to Medium",
        "season": "All season",
        "soil": "Well-drained sandy loam soil",
        "duration": "5 to 7 months",
        "summary": "Supports efficient orchard planning in warm climates with controlled irrigation cycles.",
    },
    "rice": {
        "name": "Rice",
        "water": "High",
        "season": "Kharif",
        "soil": "Clayey soil",
        "duration": "90 to 150 days",
        "summary": "Best suited to humid conditions, reliable water availability, and nutrient-rich fields.",
    },
    "watermelon": {
        "name": "Watermelon",
        "water": "Medium to High",
        "season": "Summer",
        "soil": "Sandy loam soil",
        "duration": "80 to 100 days",
        "summary": "A strong fit for warm open fields with planned irrigation and loose, well-drained soil.",
    },
}

WEATHER_SCENARIOS = [
    {
        "condition": "Warm and Humid",
        "temperature": "31 C",
        "humidity": "78%",
        "rainfall": "18 mm",
        "wind": "11 km/h",
        "summary": "Moisture is supportive for vegetative growth, but fungal risk is elevated.",
    },
    {
        "condition": "Dry Irrigation Window",
        "temperature": "34 C",
        "humidity": "42%",
        "rainfall": "3 mm",
        "wind": "16 km/h",
        "summary": "The day favors irrigation planning, mulching, and moisture conservation tasks.",
    },
    {
        "condition": "Rain-Ready Protection Window",
        "temperature": "28 C",
        "humidity": "84%",
        "rainfall": "42 mm",
        "wind": "19 km/h",
        "summary": "Field traffic and spray timing should be planned carefully due to expected rainfall.",
    },
    {
        "condition": "Cool Recovery Window",
        "temperature": "24 C",
        "humidity": "61%",
        "rainfall": "8 mm",
        "wind": "9 km/h",
        "summary": "A stable day for inspection rounds, nutrient planning, and transplant support.",
    },
]

CHATBOT_RESPONSES = [
    {
        "topic": "Crop Planning",
        "keywords": ["crop", "seed", "sowing", "variety"],
        "answer": (
            "Start crop planning by matching soil type, season, irrigation method, and NPK values. "
            "For this project, the crop intelligence form is the most reliable place to compare those conditions."
        ),
    },
    {
        "topic": "Soil Health",
        "keywords": ["soil", "organic", "compost", "microbe"],
        "answer": (
            "Healthy soil needs drainage, organic matter, and balanced nutrients. "
            "Mix compost, avoid compaction, and test pH regularly before making heavy fertilizer decisions."
        ),
    },
    {
        "topic": "Fertilizer Guidance",
        "keywords": ["fertilizer", "npk", "nitrogen", "phosphorus", "potassium"],
        "answer": (
            "Use NPK values as a starting point, then split fertilizer application across crop stages. "
            "Nitrogen supports foliage, phosphorus supports root growth, and potassium helps stress resistance."
        ),
    },
    {
        "topic": "Irrigation Planning",
        "keywords": ["irrigation", "water", "drip", "sprinkler"],
        "answer": (
            "Choose irrigation based on crop stage and soil texture. "
            "Drip irrigation is efficient for water savings, while over-irrigation can reduce root oxygen and invite disease."
        ),
    },
    {
        "topic": "Weather Awareness",
        "keywords": ["weather", "temperature", "wind", "forecast"],
        "answer": (
            "Use weather checks to plan irrigation, fertilizer timing, and spraying. "
            "Avoid chemical application before rainfall and inspect fields sooner during humid periods."
        ),
    },
    {
        "topic": "Plant Health",
        "keywords": ["plant", "leaf", "disease", "pest", "symptom", "health"],
        "answer": (
            "When leaves yellow, curl, or spot, check water stress, nutrient deficiency, and pest pressure together. "
            "Start with sanitation, scouting, and balanced irrigation before escalating treatment."
        ),
    },
    {
        "topic": "pH Management",
        "keywords": ["ph", "acidity", "alkaline"],
        "answer": (
            "Most field crops perform best when pH stays near the mildly acidic to neutral range. "
            "Very low or very high pH can lock nutrients in the soil even when fertilizer is present."
        ),
    },
    {
        "topic": "Rainfall Planning",
        "keywords": ["rain", "rainfall", "monsoon"],
        "answer": (
            "Rainfall affects sowing, drainage, pest pressure, and fertilizer efficiency. "
            "Heavy rainfall means drainage and disease prevention matter more, while low rainfall increases irrigation planning."
        ),
    },
]


DISEASE_RULES = [
    {
        "disease": "Rice Blast",
        "crop_keywords": ["rice", "paddy"],
        "symptom_keywords": ["diamond", "spindle", "grey spot", "gray spot", "brown border", "neck blast", "leaf blast", "leaf spot"],
        "field_keywords": ["humid", "rain", "wet", "dense", "cloudy"],
        "affected_parts": ["leaf", "stem", "panicle", "grain"],
        "cause": "Fungal disease favored by high humidity, wet leaves, dense planting, and poor air movement.",
        "actions": [
            "Remove badly infected plant parts where practical and avoid leaving infected residue in the field.",
            "Avoid overhead irrigation and improve spacing/air flow to reduce leaf wetness.",
            "Use a recommended fungicide only after confirming with a local agriculture officer or plant pathologist.",
        ],
        "prevention": "Use resistant varieties, balanced nitrogen, clean seed, and regular scouting during humid weather.",
    },
    {
        "disease": "Bacterial Leaf Blight",
        "crop_keywords": ["rice", "paddy"],
        "symptom_keywords": ["yellowing", "water soaked", "wilt", "dry leaf edge", "leaf edge", "blight", "kresek"],
        "field_keywords": ["rain", "flood", "wind", "wet", "humid"],
        "affected_parts": ["leaf"],
        "cause": "Bacterial infection that spreads faster through splashing water, wounds, wind, and wet field conditions.",
        "actions": [
            "Avoid unnecessary field movement when leaves are wet.",
            "Remove infected debris and improve drainage where water is standing.",
            "Do not apply excess nitrogen because it can make the crop more disease-prone.",
        ],
        "prevention": "Prefer resistant varieties, clean field bunds, balanced fertilizer, and proper drainage.",
    },
    {
        "disease": "Early Blight",
        "crop_keywords": ["tomato", "potato", "brinjal", "eggplant"],
        "symptom_keywords": ["concentric", "rings", "target spot", "brown spot", "yellow halo", "lower leaves", "leaf spot"],
        "field_keywords": ["humid", "warm", "wet", "overhead"],
        "affected_parts": ["leaf", "stem", "fruit"],
        "cause": "Fungal disease often starting on older lower leaves, especially in warm and humid conditions.",
        "actions": [
            "Remove infected lower leaves carefully and keep them away from the field.",
            "Stake plants where possible and improve air movement around the canopy.",
            "Avoid watering leaves directly; water near the root zone.",
        ],
        "prevention": "Rotate crops, use mulch, avoid crowded planting, and scout lower leaves twice a week.",
    },
    {
        "disease": "Late Blight",
        "crop_keywords": ["tomato", "potato"],
        "symptom_keywords": ["black patch", "water soaked", "dark lesion", "white mold", "fast spreading", "blight", "rotting"],
        "field_keywords": ["cool", "wet", "rain", "fog", "humid"],
        "affected_parts": ["leaf", "stem", "fruit", "tuber"],
        "cause": "Aggressive disease that spreads quickly in cool, wet, and humid weather.",
        "actions": [
            "Separate infected plants or plant parts quickly to slow spread.",
            "Avoid overhead irrigation and reduce leaf wetness as much as possible.",
            "Contact a local expert quickly because late blight can damage the crop very fast.",
        ],
        "prevention": "Use disease-free seed/seedlings, maintain spacing, and monitor closely after rain or fog.",
    },
    {
        "disease": "Powdery Mildew",
        "crop_keywords": ["grapes", "mango", "cucumber", "pumpkin", "melon", "pea", "chilli", "tomato"],
        "symptom_keywords": ["white powder", "powder", "dust", "white patches", "mildew", "whitish"],
        "field_keywords": ["humid", "dry", "shade", "poor airflow", "crowded"],
        "affected_parts": ["leaf", "stem", "flower", "fruit"],
        "cause": "Fungal growth that appears like white powder on leaves and spreads faster in crowded canopies.",
        "actions": [
            "Prune or space plants to increase sunlight and airflow.",
            "Remove highly infected leaves and avoid shaking them across healthy plants.",
            "Use locally recommended sulfur or fungicide only with expert guidance.",
        ],
        "prevention": "Avoid overcrowding, reduce shade, and inspect the upper and lower leaf surfaces regularly.",
    },
    {
        "disease": "Downy Mildew",
        "crop_keywords": ["grapes", "cucumber", "pumpkin", "melon", "onion", "maize"],
        "symptom_keywords": ["yellow patches", "purple growth", "downy", "grey growth", "gray growth", "angular spots"],
        "field_keywords": ["wet", "rain", "humid", "cool", "morning dew"],
        "affected_parts": ["leaf"],
        "cause": "Disease favored by cool, wet weather and long leaf wetness periods.",
        "actions": [
            "Improve drainage and avoid evening irrigation.",
            "Remove infected leaves early and keep the canopy open.",
            "Plan protective spray only based on local agriculture expert advice.",
        ],
        "prevention": "Use proper spacing, avoid wet foliage for long periods, and scout after rainy nights.",
    },
    {
        "disease": "Anthracnose",
        "crop_keywords": ["mango", "chilli", "papaya", "banana", "beans", "grapes"],
        "symptom_keywords": ["sunken spots", "black spots", "dark spots", "fruit rot", "lesions", "anthracnose"],
        "field_keywords": ["rain", "humid", "wet", "warm"],
        "affected_parts": ["leaf", "fruit", "stem", "flower"],
        "cause": "Fungal disease that commonly affects leaves, flowers, and fruits during humid or rainy periods.",
        "actions": [
            "Remove infected fruits/leaves and avoid dumping them near healthy plants.",
            "Improve field sanitation and avoid overhead irrigation where possible.",
            "Harvest and handle fruits carefully to avoid wounds that invite infection.",
        ],
        "prevention": "Maintain field hygiene, prune for airflow, and scout during flowering and fruiting stages.",
    },
    {
        "disease": "Leaf Curl Virus / Sucking Pest Damage",
        "crop_keywords": ["cotton", "tomato", "chilli", "papaya", "okra"],
        "symptom_keywords": ["leaf curl", "curling", "twisted", "stunted", "mosaic", "yellow vein", "small leaves"],
        "field_keywords": ["whitefly", "aphid", "pest", "dry", "hot"],
        "affected_parts": ["leaf", "whole plant"],
        "cause": "Often linked with virus spread by sucking pests such as whiteflies or aphids.",
        "actions": [
            "Inspect the underside of leaves for whiteflies, aphids, and other sucking pests.",
            "Remove severely infected plants if the problem is localized and spreading.",
            "Use yellow sticky traps and local pest-management advice before spraying.",
        ],
        "prevention": "Control weeds, monitor vectors early, use healthy seedlings, and avoid uncontrolled pesticide use.",
    },
    {
        "disease": "Rust Disease",
        "crop_keywords": ["wheat", "maize", "beans", "coffee", "groundnut", "soybean"],
        "symptom_keywords": ["rust", "orange powder", "brown powder", "pustules", "reddish spots"],
        "field_keywords": ["humid", "dew", "cool", "dense"],
        "affected_parts": ["leaf", "stem"],
        "cause": "Fungal disease that forms powder-like pustules and spreads through spores.",
        "actions": [
            "Avoid touching infected plants and then healthy plants during wet conditions.",
            "Improve airflow and remove heavily infected crop residue after harvest.",
            "Confirm severity with a local expert if pustules spread quickly.",
        ],
        "prevention": "Use resistant varieties, rotate crops, and scout early when dew and humidity are high.",
    },
    {
        "disease": "Root Rot / Damping Off",
        "crop_keywords": ["tomato", "chilli", "cotton", "beans", "groundnut", "vegetable", "nursery"],
        "symptom_keywords": ["wilting", "root rot", "black root", "soft stem", "seedling death", "damping", "collar rot"],
        "field_keywords": ["waterlogged", "overwater", "poor drainage", "wet soil"],
        "affected_parts": ["root", "stem", "whole plant"],
        "cause": "Soil-borne problem favored by excess moisture, poor drainage, and weak root-zone aeration.",
        "actions": [
            "Stop excess watering and improve drainage immediately.",
            "Remove dead seedlings/plants and avoid reusing infected nursery soil.",
            "Add organic matter and avoid compacted soil around the root zone.",
        ],
        "prevention": "Use raised beds, treated/healthy seed, clean nursery trays, and proper irrigation scheduling.",
    },
]

AFFECTED_PART_OPTIONS = ["Leaf", "Stem", "Root", "Flower", "Fruit", "Whole plant"]
FIELD_CONDITION_OPTIONS = [
    "Humid / rainy",
    "Dry / hot",
    "Waterlogged soil",
    "Poor airflow / crowded crop",
    "Pest visible",
    "Normal field condition",
]
SEVERITY_OPTIONS = ["Mild", "Moderate", "Severe / spreading fast"]
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_image_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _keyword_hits(text, keywords):
    return [keyword for keyword in keywords if keyword.lower() in text]


def build_disease_result(form_values):
    crop = form_values.get("crop", "").strip()
    symptoms = form_values.get("symptoms", "").strip()
    affected_part = form_values.get("affected_part", "").strip()
    field_condition = form_values.get("field_condition", "").strip()
    severity = form_values.get("severity", "").strip()

    combined_text = f"{crop} {symptoms} {affected_part} {field_condition} {severity}".lower()
    ranked_matches = []

    for rule in DISEASE_RULES:
        crop_hits = _keyword_hits(crop.lower(), rule["crop_keywords"])
        symptom_hits = _keyword_hits(combined_text, rule["symptom_keywords"])
        field_hits = _keyword_hits(combined_text, rule["field_keywords"])
        part_hits = _keyword_hits(affected_part.lower(), rule["affected_parts"])

        score = (len(crop_hits) * 3) + (len(symptom_hits) * 4) + (len(field_hits) * 2) + len(part_hits)

        if score > 0:
            ranked_matches.append(
                {
                    "score": score,
                    "rule": rule,
                    "matched_keywords": list(dict.fromkeys(crop_hits + symptom_hits + field_hits + part_hits)),
                }
            )

    ranked_matches.sort(key=lambda item: item["score"], reverse=True)

    if not ranked_matches:
        return {
            "disease": "General Plant Stress / Unknown Symptom Pattern",
            "confidence": "Low",
            "cause": "The entered symptoms do not strongly match the built-in disease rules. It may be nutrient stress, water imbalance, pest attack, or an early disease stage.",
            "matched_keywords": [],
            "actions": [
                "Take clear photos of affected leaves, stem, root area, and the full plant.",
                "Check soil moisture, recent fertilizer use, pest presence, and whether symptoms are spreading.",
                "Ask a local agriculture officer or plant doctor before applying pesticide or fungicide.",
            ],
            "prevention": "Keep weekly records of symptoms, irrigation, fertilizer, weather, and pest sightings.",
            "disclaimer": "This is a rule-based demo advisory, not a laboratory diagnosis.",
        }

    best = ranked_matches[0]
    rule = best["rule"]
    score = best["score"]

    if score >= 13:
        confidence = "High"
    elif score >= 7:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "disease": rule["disease"],
        "confidence": confidence,
        "cause": rule["cause"],
        "matched_keywords": best["matched_keywords"],
        "actions": rule["actions"],
        "prevention": rule["prevention"],
        "disclaimer": "This is a symptom-based demo finder. Confirm with a local agriculture expert before chemical treatment.",
    }




def analyze_crop_photo(image_path, crop_name=""):
    """Photo-based demo disease finder using simple color/spot rules.

    This is not a trained CNN model. It helps the project accept a crop photo
    and gives possible disease/stress guidance from visible color patterns.
    """
    if Image is None:
        return {
            "disease": "Image analysis package missing",
            "confidence": "Not available",
            "visual_signs": ["Pillow is not installed"],
            "actions": [
                "Install Pillow using: pip install pillow",
                "Then restart Flask using: python app.py",
            ],
            "prevention": "After installing Pillow, upload a clear leaf/crop photo again.",
            "disclaimer": "The image could not be processed because Pillow is missing.",
            "image_url": None,
        }

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((360, 360))
        pixels = list(image.getdata())

    total = max(len(pixels), 1)

    green = yellow = brown = white = dark = 0
    for red, green_value, blue in pixels:
        max_channel = max(red, green_value, blue)
        min_channel = min(red, green_value, blue)

        if green_value > 70 and green_value > red * 1.12 and green_value > blue * 1.12:
            green += 1
        if red > 115 and green_value > 95 and blue < 120 and abs(red - green_value) < 95:
            yellow += 1
        if 45 < red < 175 and 20 < green_value < 135 and blue < 115 and red >= green_value * 0.85:
            brown += 1
        if red > 185 and green_value > 185 and blue > 185 and (max_channel - min_channel) < 55:
            white += 1
        if red < 70 and green_value < 70 and blue < 70:
            dark += 1

    percentages = {
        "green": round((green / total) * 100, 1),
        "yellow": round((yellow / total) * 100, 1),
        "brown_spots": round((brown / total) * 100, 1),
        "white_powder": round((white / total) * 100, 1),
        "dark_rotten_area": round((dark / total) * 100, 1),
    }

    crop = crop_name.lower().strip()
    disease = "Possible Plant Stress / Needs Manual Check"
    confidence = "Low"
    visual_signs = []
    actions = [
        "Take one close-up leaf photo and one full-plant photo in daylight.",
        "Check underside of leaves for insects, eggs, webbing, or sticky honeydew.",
        "Avoid spraying chemicals until the disease is confirmed by a local agriculture expert.",
    ]
    prevention = "Maintain balanced irrigation, remove infected residue, avoid overcrowding, and monitor the crop every 3 to 4 days."

    if percentages["green"] >= 45 and percentages["yellow"] < 12 and percentages["brown_spots"] < 10 and percentages["white_powder"] < 8:
        disease = "Mostly Healthy / No Strong Disease Pattern Detected"
        confidence = "Medium"
        visual_signs = ["High green leaf area", "Low visible yellowing", "Low visible spot/rot area"]
        actions = [
            "Continue weekly plant health monitoring.",
            "Check leaves again after 3 to 5 days if symptoms are starting.",
            "Keep proper spacing, irrigation, and nutrient balance.",
        ]
        prevention = "Continue preventive scouting, remove weak leaves, and avoid overwatering."
    elif percentages["white_powder"] >= 8:
        disease = "Possible Powdery Mildew / Fungal Growth"
        confidence = "Medium"
        visual_signs = ["White powder-like bright patches detected", "Possible fungal surface growth"]
        actions = [
            "Remove badly infected leaves where practical.",
            "Improve airflow and avoid overhead watering.",
            "Show the photo to a local agriculture officer before fungicide use.",
        ]
        prevention = "Use spacing, pruning, morning irrigation, and resistant varieties where available."
    elif percentages["brown_spots"] >= 14 or (percentages["brown_spots"] >= 8 and percentages["yellow"] >= 12):
        if "rice" in crop:
            disease = "Possible Rice Blast / Leaf Spot / Bacterial Blight Pattern"
        elif "tomato" in crop:
            disease = "Possible Tomato Early Blight / Leaf Spot Pattern"
        elif "potato" in crop:
            disease = "Possible Potato Blight / Leaf Spot Pattern"
        else:
            disease = "Possible Leaf Spot / Blight Disease Pattern"
        confidence = "Medium"
        visual_signs = ["Brown/dark spot area detected", "Yellowing around affected area may be present"]
        actions = [
            "Remove heavily infected leaves and do not leave them in the field.",
            "Avoid wetting leaves during irrigation.",
            "Improve spacing and airflow to reduce fungal spread.",
            "Confirm locally before applying fungicide or bactericide.",
        ]
        prevention = "Use crop rotation, clean tools, field sanitation, and avoid dense planting."
    elif percentages["yellow"] >= 22:
        if "cotton" in crop:
            disease = "Possible Yellowing / Leaf Curl / Nutrient or Pest Stress"
        elif "rice" in crop:
            disease = "Possible Rice Yellowing / Nutrient Stress / Disease Pressure"
        else:
            disease = "Possible Yellowing / Chlorosis / Nutrient or Disease Stress"
        confidence = "Medium"
        visual_signs = ["High yellow leaf area detected", "Possible chlorosis or stress"]
        actions = [
            "Check soil moisture first: both overwatering and underwatering can cause yellowing.",
            "Check for sucking pests under the leaves.",
            "Review nitrogen and micronutrient supply.",
            "If yellowing is spreading fast, ask a local plant doctor/agriculture officer.",
        ]
        prevention = "Use balanced fertilizer, proper drainage, pest monitoring, and avoid sudden irrigation stress."
    elif percentages["dark_rotten_area"] >= 12:
        disease = "Possible Rot / Necrosis / Severe Tissue Damage"
        confidence = "Low to Medium"
        visual_signs = ["Dark rotten/necrotic area detected", "Possible advanced damage"]
        actions = [
            "Remove dead or rotting plant parts safely.",
            "Check root zone drainage and smell for root rot signs.",
            "Do not reuse infected nursery soil.",
            "Get local expert confirmation quickly if spread is fast.",
        ]
        prevention = "Improve drainage, avoid waterlogging, use healthy seedlings, and sanitize tools."
    else:
        visual_signs = [
            "No strong disease color pattern detected",
            "The photo may be unclear, too dark, or symptoms may be early stage",
        ]

    image_name = Path(image_path).name
    return {
        "disease": disease,
        "confidence": confidence,
        "visual_signs": visual_signs,
        "percentages": percentages,
        "actions": actions,
        "prevention": prevention,
        "disclaimer": "This is photo-based demo analysis, not a trained laboratory diagnosis. For accurate disease detection, use a trained CNN/PlantVillage model or consult a local agriculture expert.",
        "image_url": url_for("static", filename=f"uploads/{image_name}"),
    }

def get_user_profile():
    profile = DEMO_USER.copy()
    profile.update(session.get("user", {}))
    return profile


def render_page(template_name, page_key, page_title, page_subtitle, page_eyebrow, **context):
    return render_template(
        template_name,
        page_key=page_key,
        page_title=page_title,
        page_subtitle=page_subtitle,
        page_eyebrow=page_eyebrow,
        **context,
    )


def build_weather_report(location):
    cleaned = "".join(character for character in location.lower() if character.isalnum())
    scenario = WEATHER_SCENARIOS[sum(ord(character) for character in cleaned) % len(WEATHER_SCENARIOS)].copy()

    tips = []
    condition = scenario["condition"]

    if "Humid" in condition or "Rain" in condition:
        tips.append("Increase field scouting for leaf spot, mildew, and other humidity-driven disease pressure.")
    if "Dry" in condition:
        tips.append("Prioritize early-morning irrigation and mulch exposed soil to reduce evaporation losses.")
    if "Cool" in condition:
        tips.append("Use the stable window for nutrient correction, transplant establishment, and root-zone checks.")

    tips.extend(
        [
            "Avoid fertilizer spraying immediately before rainfall or strong wind windows.",
            "Check drainage channels and irrigation lines before the next field operation cycle.",
        ]
    )

    operations = [
        {
            "title": "Water Planning",
            "description": "Align irrigation timing with the current moisture outlook to reduce stress and runoff.",
        },
        {
            "title": "Crop Protection",
            "description": "Use the humidity and rainfall pattern to decide when disease scouting should be intensified.",
        },
        {
            "title": "Field Operations",
            "description": "Schedule fertilizer, pesticide, and intercultural work during the most stable part of the day.",
        },
    ]

    scenario.update(
        {
            "location": location.title(),
            "tips": tips,
            "operations": operations,
            "demo_note": "This advisory is generated locally as a demo and does not use an external weather API.",
        }
    )
    return scenario


def build_chatbot_reply(question):
    cleaned_question = question.strip()
    if not cleaned_question:
        return {
            "topic": "Quick Tip",
            "answer": "Ask about crops, soil, fertilizer, irrigation, weather, plant health, pH, or rainfall.",
        }

    lowered_question = cleaned_question.lower()
    for entry in CHATBOT_RESPONSES:
        if any(keyword in lowered_question for keyword in entry["keywords"]):
            return {"topic": entry["topic"], "answer": entry["answer"]}

    return {
        "topic": "General Farming Advice",
        "answer": (
            "I could not match that to a specific farming topic, so start with the basics: "
            "check soil condition, water availability, weather pattern, and crop stage before acting."
        ),
    }


def build_reason_points(crop_info, form_values):
    return [
        (
            f"The trained model matched your NPK inputs of {form_values['nitrogen']:.1f}, "
            f"{form_values['phosphorus']:.1f}, and {form_values['potassium']:.1f} "
            f"to crop conditions that closely resemble {crop_info['name']}."
        ),
        (
            f"Temperature {form_values['temperature']:.1f} C, humidity {form_values['humidity']:.1f}%, "
            f"pH {form_values['ph']:.1f}, and rainfall {form_values['rainfall']:.1f} mm create a supportive climate profile."
        ),
        (
            f"Your selected field context of {form_values['soil_type']}, {form_values['season']}, "
            f"and {form_values['irrigation']} strengthens the practical fit for this recommendation."
        ),
    ]


def build_advice_cards(crop_info, form_values):
    return [
        {
            "title": "Water Strategy",
            "description": (
                f"This crop has a {crop_info['water'].lower()} water requirement, so align irrigation intervals "
                f"with soil moisture instead of relying on calendar-based watering."
            ),
        },
        {
            "title": "Soil Management",
            "description": (
                f"Prepare the field to support {crop_info['soil'].lower()} behavior and keep pH close to the current "
                f"observed value of {form_values['ph']:.1f} unless a soil test recommends correction."
            ),
        },
        {
            "title": "Season Planning",
            "description": (
                f"Plan sowing and crop protection around the {crop_info['season'].lower()} window to maintain growth stability "
                f"through the expected duration of {crop_info['duration'].lower()}."
            ),
        },
        {
            "title": "Field Monitoring",
            "description": (
                "Scout twice a week for nutrient stress, leaf symptoms, and water imbalance so corrective action starts early."
            ),
        },
    ]


def format_input_conditions(form_values):
    return [
        {"label": "Nitrogen", "value": f"{form_values['nitrogen']:.1f}"},
        {"label": "Phosphorus", "value": f"{form_values['phosphorus']:.1f}"},
        {"label": "Potassium", "value": f"{form_values['potassium']:.1f}"},
        {"label": "Temperature", "value": f"{form_values['temperature']:.1f} C"},
        {"label": "Humidity", "value": f"{form_values['humidity']:.1f}%"},
        {"label": "pH Value", "value": f"{form_values['ph']:.1f}"},
        {"label": "Rainfall", "value": f"{form_values['rainfall']:.1f} mm"},
    ]


def build_profile_from_form(form_data):
    profile = get_user_profile()
    profile.update(
        {
            "name": form_data.get("name", profile["name"]).strip() or profile["name"],
            "email": form_data.get("email", profile["email"]).strip() or profile["email"],
            "phone": form_data.get("phone", profile["phone"]).strip() or profile["phone"],
            "location": form_data.get("location", profile["location"]).strip() or profile["location"],
            "land_area": form_data.get("land_area", profile["land_area"]).strip() or profile["land_area"],
            "experience": form_data.get("experience", profile["experience"]).strip() or profile["experience"],
            "focus_crop": form_data.get("focus_crop", profile["focus_crop"]).strip() or profile["focus_crop"],
        }
    )
    return profile


@app.context_processor
def inject_layout_context():
    return {
        "app_name": APP_NAME,
        "app_tagline": APP_TAGLINE,
        "nav_items": NAV_ITEMS,
        "current_user": get_user_profile(),
        "is_demo_user": "user" not in session,
    }


@app.route("/")
def home():
    return render_page(
        "home.html",
        page_key="home",
        page_title="Smart farming decisions with a product-grade experience",
        page_subtitle="Use your existing Flask and ML stack to guide crop planning, field health, weather action, and day-to-day advisory support.",
        page_eyebrow="AgriVision AI Platform",
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email or not password:
            flash("Enter both email and password to open the demo workspace.", "error")
            return redirect(url_for("login"))

        derived_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
        profile = get_user_profile()
        profile.update({"email": email, "name": derived_name or profile["name"]})
        session["user"] = profile

        flash("Demo login successful. Your smart farming dashboard is ready.", "success")
        return redirect(url_for("dashboard"))

    return render_page(
        "login.html",
        page_key="login",
        page_title="Access the farmer workspace",
        page_subtitle="This project uses a stable demo login flow because no database authentication is configured in the uploaded stack.",
        page_eyebrow="Demo Access",
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        profile = build_profile_from_form(request.form)
        session["user"] = profile
        flash("Demo account created. You can continue directly to the dashboard.", "success")
        return redirect(url_for("dashboard"))

    return render_page(
        "register.html",
        page_key="register",
        page_title="Create a polished demo farmer profile",
        page_subtitle="Profile details are stored only in the session so the UI stays functional without adding a new database.",
        page_eyebrow="Demo Onboarding",
    )


@app.route("/dashboard")
def dashboard():
    return render_page(
        "dashboard.html",
        page_key="dashboard",
        page_title="Farm command center",
        page_subtitle="Move from crop recommendation to advisory workflows through one clean, presentation-ready smart farming dashboard.",
        page_eyebrow="Operations Dashboard",
        last_prediction=session.get("last_prediction"),
    )


@app.route("/crop-intelligence")
def crop_intelligence():
    profile = get_user_profile()
    form_defaults = {
        "farmer_name": profile["name"],
        "phone": profile["phone"],
        "location": profile["location"],
        "land_area": profile["land_area"],
    }

    return render_page(
        "crop_intelligence.html",
        page_key="crop_intelligence",
        page_title="Crop Intelligence",
        page_subtitle="Submit nutrient, climate, and field context to the existing trained model and receive a polished recommendation result.",
        page_eyebrow="ML Crop Recommendation",
        soil_options=SOIL_OPTIONS,
        season_options=SEASON_OPTIONS,
        irrigation_options=IRRIGATION_OPTIONS,
        form_defaults=form_defaults,
    )


@app.route("/predict", methods=["POST"])
def predict():
    profile = get_user_profile()
    try:
        form_values = {
            "farmer_name": request.form.get("farmer_name", profile["name"]).strip() or profile["name"],
            "phone": request.form.get("phone", profile["phone"]).strip() or profile["phone"],
            "location": request.form.get("location", profile["location"]).strip() or profile["location"],
            "land_area": request.form.get("land_area", profile["land_area"]).strip() or profile["land_area"],
            "soil_type": request.form.get("soil_type", "").strip(),
            "season": request.form.get("season", "").strip(),
            "irrigation": request.form.get("irrigation", "").strip(),
            "nitrogen": float(request.form.get("nitrogen", "")),
            "phosphorus": float(request.form.get("phosphorus", "")),
            "potassium": float(request.form.get("potassium", "")),
            "temperature": float(request.form.get("temperature", "")),
            "humidity": float(request.form.get("humidity", "")),
            "ph": float(request.form.get("ph", "")),
            "rainfall": float(request.form.get("rainfall", "")),
        }
    except ValueError:
        flash("Enter valid numeric values for all crop intelligence fields.", "error")
        return redirect(url_for("crop_intelligence"))

    if not all([form_values["soil_type"], form_values["season"], form_values["irrigation"]]):
        flash("Select soil type, season, and irrigation type before running the prediction.", "error")
        return redirect(url_for("crop_intelligence"))

    input_data = pd.DataFrame(
        [[
            form_values["nitrogen"],
            form_values["phosphorus"],
            form_values["potassium"],
            form_values["temperature"],
            form_values["humidity"],
            form_values["ph"],
            form_values["rainfall"],
        ]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )

    try:
        prediction = model.predict(input_data)[0]
    except Exception:
        flash("The crop model could not complete the prediction. Please try again.", "error")
        return redirect(url_for("crop_intelligence"))

    crop_info = CROP_DETAILS.get(
        prediction,
        {
            "name": str(prediction).replace("_", " ").title(),
            "water": "Not available",
            "season": "Not available",
            "soil": "Not available",
            "duration": "Not available",
            "summary": "The trained model returned a crop label, but no additional advisory metadata was configured for it.",
        },
    )

    session["last_prediction"] = {
        "crop": crop_info["name"],
        "season": crop_info["season"],
        "water": crop_info["water"],
    }

    return render_page(
        "result.html",
        page_key="crop_intelligence",
        page_title="Prediction Result",
        page_subtitle="The existing trained model has completed its recommendation using the submitted field and climate profile.",
        page_eyebrow="Result Ready",
        prediction_label=crop_info["name"],
        crop_info=crop_info,
        reason_points=build_reason_points(crop_info, form_values),
        advice_cards=build_advice_cards(crop_info, form_values),
        input_conditions=format_input_conditions(form_values),
        field_context=[
            {"label": "Farm Location", "value": form_values["location"]},
            {"label": "Land Area", "value": form_values["land_area"]},
            {"label": "Selected Soil Type", "value": form_values["soil_type"]},
            {"label": "Selected Season", "value": form_values["season"]},
            {"label": "Irrigation Type", "value": form_values["irrigation"]},
        ],
    )


@app.route("/plant-health", methods=["GET", "POST"])
def plant_health():
    disease_result = None
    image_result = None
    disease_form = {
        "crop": "",
        "affected_part": "",
        "field_condition": "",
        "severity": "",
        "symptoms": "",
    }
    photo_form = {
        "image_crop": "",
    }

    if request.method == "POST":
        check_type = request.form.get("check_type", "symptom")

        if check_type == "image":
            photo_form = {
                "image_crop": request.form.get("image_crop", "").strip(),
            }
            crop_photo = request.files.get("crop_photo")

            if not crop_photo or not crop_photo.filename:
                flash("Upload a clear crop or leaf photo first.", "error")
                return redirect(url_for("plant_health"))

            if not allowed_image_file(crop_photo.filename):
                flash("Only PNG, JPG, JPEG, and WEBP image files are allowed.", "error")
                return redirect(url_for("plant_health"))

            safe_name = secure_filename(crop_photo.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            save_path = app.config["UPLOAD_FOLDER"] / unique_name
            crop_photo.save(save_path)

            image_result = analyze_crop_photo(save_path, photo_form["image_crop"])

        else:
            disease_form = {
                "crop": request.form.get("crop", "").strip(),
                "affected_part": request.form.get("affected_part", "").strip(),
                "field_condition": request.form.get("field_condition", "").strip(),
                "severity": request.form.get("severity", "").strip(),
                "symptoms": request.form.get("symptoms", "").strip(),
            }

            if not disease_form["crop"] or not disease_form["symptoms"]:
                flash("Enter crop name and visible symptoms to find possible disease guidance.", "error")
                return redirect(url_for("plant_health"))

            disease_result = build_disease_result(disease_form)

    return render_page(
        "plant_health.html",
        page_key="plant_health",
        page_title="Plant Health & Disease Finder",
        page_subtitle="Upload a crop photo or enter visible symptoms to find possible disease patterns and safe farmer-friendly actions.",
        page_eyebrow="Crop Disease Support",
        disease_result=disease_result,
        image_result=image_result,
        disease_form=disease_form,
        photo_form=photo_form,
        affected_part_options=AFFECTED_PART_OPTIONS,
        field_condition_options=FIELD_CONDITION_OPTIONS,
        severity_options=SEVERITY_OPTIONS,
    )

@app.route("/weather", methods=["GET", "POST"])
def weather():
    weather_report = None
    location = ""

    if request.method == "POST":
        location = request.form.get("location", "").strip()
        if not location:
            flash("Enter a city or farm location to generate the advisory.", "error")
            return redirect(url_for("weather"))

        weather_report = build_weather_report(location)

    return render_page(
        "weather.html",
        page_key="weather",
        page_title="Weather Advisory",
        page_subtitle="Generate a stable local demo weather summary and farming action list without depending on an external API key.",
        page_eyebrow="Demo Advisory Engine",
        weather_report=weather_report,
        submitted_location=location,
    )


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    question = ""
    response = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        response = build_chatbot_reply(question)

    return render_page(
        "chatbot.html",
        page_key="chatbot",
        page_title="AI Chatbot",
        page_subtitle="Use the built-in rule-based farming assistant for common questions on crop, soil, fertilizer, irrigation, weather, plant health, pH, and rainfall.",
        page_eyebrow="Rule-Based Assistant",
        question=question,
        response=response,
    )


@app.route("/profile")
def profile():
    return render_page(
        "profile.html",
        page_key="profile",
        page_title="Farmer Profile",
        page_subtitle="View and manage the active farmer details used across the crop recommendation workflow.",
        page_eyebrow="Profile Workspace",
        last_prediction=session.get("last_prediction"),
    )


@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():
    if request.method == "POST":
        updated_profile = build_profile_from_form(request.form)
        session["user"] = updated_profile
        flash("Farmer profile updated successfully.", "success")
        return redirect(url_for("profile"))

    return render_page(
        "edit_profile.html",
        page_key="profile",
        page_title="Edit Farmer Profile",
        page_subtitle="Update farmer, farm, and crop details. These details will be used to prefill the crop intelligence workflow.",
        page_eyebrow="Edit Profile",
    )


@app.route("/about")
def about():
    return render_page(
        "about.html",
        page_key="about",
        page_title="About AgriVision AI",
        page_subtitle="This project keeps the existing Flask application and ML prediction flow while packaging the experience like a real smart farming product.",
        page_eyebrow="Platform Overview",
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Demo session cleared.", "info")
    return redirect(url_for("login"))


@app.route("/farmer")
def farmer():
    return redirect(url_for("crop_intelligence"))


@app.route("/crop", methods=["GET", "POST"])
def crop():
    return redirect(url_for("crop_intelligence"))


@app.route("/disease", methods=["GET", "POST"])
def disease():
    return redirect(url_for("plant_health"))


if __name__ == "__main__":
    app.run(debug=True)
