import json
import re
import random
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = "roamie_production_key_fixed"

STATE_AWAITING_NAME = "AWAITING_NAME"
STATE_IDLE = "IDLE"

MUNICIPALITIES = [
    "laoag", "batac", "san nicolas", "bacarra", "paoay", 
    "currimao", "dingras", "pagudpud", "bangui", "burgos", 
    "sarrat", "vintar", "piddig", "pasuquin", "solsona", "badoc"
]

CATEGORY_SYNONYMS = {
    "restaurant": ["restaurant", "bistro", "eatery", "dining", "grill", "steak", "steaks", "food", "wings", "pasta", "sushi", "ramen", "seafood", "samgyupsal", "samgyup", "korean bbq", "bbq", "eat", "fastfood"],
    "cafe": ["cafe", "coffee", "bakeshop", "bakery", "espresso", "latte", "tea"],
    "karaoke": ["karaoke", "ktv", "singing", "music lounge"],
    "gym": ["gym", "fitness", "workout"],
    "hotel": ["hotel", "resort", "inn", "lodging", "stay"],
    "hospital": ["hospital", "medical center", "clinic", "health center", "infirmary"],
    "bank": ["bank", "atm"],
    "gas station": ["gas station", "gasoline station", "gasoline", "gas", "fuel", "petrol"]
}

THINKING_PHRASES = [
    "still thinking", "wait", "hold on", "give me a sec", 
    "give me a minute", "thinking", "not sure yet", "wait a bit"
]

READY_PHRASES = ["im ready", "i'm ready", "ready", "ok ready", "okay ready"]

STOP_WORDS = {
    "want", "place", "places", "that", "open", "until", "have", "has", "with", "find", 
    "looking", "near", "around", "show", "list", "under", "below", "the", "in", 
    "city", "whats", "what", "is", "are", "nearest", "im", "ready", "you", 
    "where", "can", "get", "there", "for", "please", "some", "any", "they", 
    "does", "do", "serve", "serving", "okay", "ok", "at", "here", "recommend", "me", "eat", "all"
}

GREETINGS = {"hello", "hi", "hey", "sup", "good morning", "good afternoon", "good evening"}

def load_dataset():
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset.json: {e}")
        return {}

dataset = load_dataset()

def get_all_places():
    if isinstance(dataset, dict):
        return [place for city_places in dataset.values() for place in city_places]
    return dataset if isinstance(dataset, list) else []

def get_place_attr(place, keys, default="N/A"):
    for key in keys:
        if key in place and place[key]:
            return place[key]
    return default

def get_user_session():
    if "user_data" not in session:
        session["user_data"] = {
            "name": None,
            "location": None,
            "state": STATE_AWAITING_NAME,
            "last_places": [],
            "last_category": None
        }
    return session["user_data"]

def normalize_and_tokenize(text):
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = clean_text.split()
    return clean_text, tokens

def parse_time_to_24h(time_str, am_pm):
    hour = int(time_str)
    am_pm = am_pm.upper()
    if am_pm == "PM" and hour < 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0
    return hour

def extract_constraints(query_text):
    clean_text, tokens = normalize_and_tokenize(query_text)
    
    constraints = {
        "category": None,
        "location": None,
        "closing_time_req": None,
        "max_budget": None,
        "negations": [],
        "features": []
    }

    constraints["negations"] = re.findall(r'\b(?:no|not|without)\s+([a-z]+)\b', clean_text)

    # Location Extraction
    for muni in MUNICIPALITIES:
        if re.search(rf"\b{re.escape(muni)}\b", clean_text):
            constraints["location"] = muni
            break

    # Category Match
    for cat, syns in CATEGORY_SYNONYMS.items():
        for syn in syns:
            if re.search(rf"\b{re.escape(syn)}\b", clean_text):
                constraints["category"] = cat
                break
        if constraints["category"]:
            break

    # Closing Time Match
    time_match = re.search(r'(?:until|open until|at|till|closing at)\s*(\d{1,2})\s*(:\d{2})?\s*(am|pm)', clean_text)
    if time_match:
        constraints["closing_time_req"] = parse_time_to_24h(time_match.group(1), time_match.group(3))

    # Budget Match
    budget_match = re.search(r'(?:under|below|less than|max|budget of)\s*(?:₱|p)?\s*(\d+)', clean_text)
    if budget_match:
        constraints["max_budget"] = int(budget_match.group(1))

    # Features (Retain specific food/item queries like "salmon", "cozy", "wifi")
    for token in tokens:
        if (token not in STOP_WORDS and 
            token not in MUNICIPALITIES and 
            token not in constraints["negations"] and 
            len(token) >= 3):
            constraints["features"].append(token)

    return constraints

def score_and_rank_places(constraints, session_location=None):
    all_places = get_all_places()
    scored_places = []

    req_category = constraints.get("category")
    req_location = constraints.get("location") or session_location
    req_features = constraints.get("features", [])

    category_synonyms = CATEGORY_SYNONYMS.get(req_category, [req_category]) if req_category else []

    # Get non-category specific words (e.g., 'salmon')
    specific_features = [
        f for f in req_features 
        if f not in category_synonyms and f not in [req_category]
    ]

    for place in all_places:
        p_name = str(get_place_attr(place, ["Business name", "name", "business_name"], "")).lower()
        p_cat = str(get_place_attr(place, ["Category", "category", "Type"], "")).lower()
        p_city = str(get_place_attr(place, ["Municipality or city", "city", "municipality", "Location"], "")).lower()
        
        p_feats = [str(f).lower() for f in get_place_attr(place, ["Features", "features"], [])]
        p_desc = str(get_place_attr(place, ["Short description", "Description", "description"], "")).lower()
        p_keywords = [str(k).lower() for k in get_place_attr(place, ["Keywords and synonyms", "Keywords"], [])]
        
        full_text = f"{p_name} {p_cat} {p_desc} {' '.join(p_feats)} {' '.join(p_keywords)}"

        # 1. HARD LOCATION FILTER
        if req_location:
            clean_req = req_location.lower().replace("city", "").strip()
            clean_p = p_city.lower().replace("city", "").strip()
            if clean_req not in clean_p:
                continue

        # 2. HARD CATEGORY FILTER
        if req_category:
            is_cat_match = any(re.search(rf"\b{re.escape(syn)}\b", full_text) for syn in category_synonyms)
            if not is_cat_match:
                continue

        # 3. STRICT FEATURE MATCH FOR SPECIFIC ITEMS (e.g., "salmon")
        if specific_features:
            if not any(feat in full_text for feat in specific_features):
                continue  # Drop places that don't explicitly mention salmon

        # SCORING
        score = 100
        for feat in req_features:
            if feat in full_text:
                score += 25

        scored_places.append({"place": place, "score": score})

    # Shuffle first to break alphabetical ties (so gas stations are randomized)
    random.shuffle(scored_places)
    # Sort by match score
    scored_places.sort(key=lambda x: x["score"], reverse=True)
    return scored_places

def format_place_output(ranked_results, user_name, has_category=True):
    top_results = ranked_results[:3]
    if not top_results:
        return f"I couldn't find any places matching all your criteria, {user_name}."

    # Dynamic Header based on query type
    if has_category:
        reply = f"Here are the top options matching your query, {user_name}:<br><br>"
    else:
        reply = f"Here are my suggested places for you, {user_name}:<br><br>"

    for idx, item in enumerate(top_results, 1):
        place = item["place"]
        name = get_place_attr(place, ["Business name", "name", "business_name"])
        cat = get_place_attr(place, ["Category", "category"])
        city = get_place_attr(place, ["Municipality or city", "city", "municipality"])
        hours = get_place_attr(place, ["Opening and closing hours", "Operating Hours", "hours"])
        price = get_place_attr(place, ["Price range", "Price Range", "price_range"])

        reply += f"<b>{idx}. {name}</b> ({cat})<br>"
        reply += f"📍 <b>Location</b>: {city}<br>"
        reply += f"⏰ <b>Hours</b>: {hours}<br>"
        reply += f"💰 <b>Price</b>: {price}<br>"

        if idx < len(top_results):
            reply += "<hr style='border: 0; border-top: 1px solid #444; margin: 10px 0;'>"

    if len(top_results) == 1:
        reply += "<br><b>Which one suits you?</b> (Reply with 1 to view description and full details!)"
    else:
        reply += f"<br><b>Which one suits you?</b> (Reply with 1 to {len(top_results)} to view description and full details!)"
    
    return reply

def format_single_place_detail(item, user_name):
    place = item.get("place", item) if isinstance(item, dict) and "place" in item else item
    
    name = get_place_attr(place, ["Business name", "name", "business_name"])
    cat = get_place_attr(place, ["Category", "category"])
    city = get_place_attr(place, ["Municipality or city", "city", "municipality"])
    hours = get_place_attr(place, ["Opening and closing hours", "Operating Hours", "hours"])
    price = get_place_attr(place, ["Price range", "Price Range", "price_range"])
    
    features_list = get_place_attr(place, ["Features", "features"], default=[])
    features = ", ".join(features_list) if isinstance(features_list, list) else features_list
    description = get_place_attr(place, ["Short description", "Description", "description"], default="No description available.")

    reply = f"Here are the full details for <b>{name}</b>, {user_name}:<br><br>"
    reply += f"🏢 <b>Category</b>: {cat}<br>"
    reply += f"📍 <b>Location</b>: {city}<br>"
    reply += f"⏰ <b>Hours</b>: {hours}<br>"
    reply += f"💰 <b>Price Range</b>: {price}<br>"
    if features: 
        reply += f"✨ <b>Features</b>: {features}<br>"
    reply += f"<br>📝 <b>Description</b>:<br>{description}"
    return reply

@app.route("/")
def home():
    session["user_data"] = {
        "name": None,
        "location": None,
        "state": STATE_AWAITING_NAME,
        "last_places": [],
        "last_category": None
    }
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    msg = data.get("message", "").strip()
    clean_text, tokens = normalize_and_tokenize(msg)
    msg_lower = clean_text
    
    u_session = get_user_session()

    # Name Capture Phase
    if u_session["state"] == STATE_AWAITING_NAME or u_session["name"] is None:
        clean_name = re.sub(
            r'^(hello|hi|hey)?\s*,?\s*(my name is|call me|im|i\'m|i am)\s*', 
            '', 
            msg, 
            flags=re.IGNORECASE
        ).strip()
        clean_name = re.sub(r'^(hello|hi|hey)\s+', '', clean_name, flags=re.IGNORECASE).strip()
        
        u_session["name"] = clean_name.title() if clean_name else "Friend"
        u_session["state"] = STATE_IDLE
        session.modified = True
        return jsonify({
            "reply": f"Nice to meet you, {u_session['name']}! What kind of place are you looking for in Ilocos Norte today?"
        })

    display_name = u_session["name"]

    # Selection Phase (1, 2, 3)
    if u_session["last_places"] and (msg_lower in ["1", "2", "3", "option 1", "option 2", "option 3", "first", "second", "third"]):
        idx_map = {"1": 0, "option 1": 0, "first": 0, "2": 1, "option 2": 1, "second": 1, "3": 2, "option 3": 2, "third": 2}
        selected_idx = idx_map.get(msg_lower)
        if selected_idx is not None and selected_idx < len(u_session["last_places"]):
            chosen_place = u_session["last_places"][selected_idx]
            return jsonify({"reply": format_single_place_detail(chosen_place, display_name)})

    if msg_lower in ["okay", "ok"]:
        return jsonify({"reply": f"Got it, {display_name}! What place or category would you like to check out next?"})

    if any(phrase in msg_lower for phrase in THINKING_PHRASES):
        return jsonify({
            "reply": f"No rush at all, {display_name}! Take your time. Just let me know whenever you're ready to search."
        })

    if any(phrase in msg_lower for phrase in READY_PHRASES):
        return jsonify({
            "reply": f"Awesome! What kind of place or category would you like to search for, {display_name}?"
        })

    if msg_lower in GREETINGS:
        return jsonify({
            "reply": f"Hello {display_name}! How can I help you explore Ilocos Norte right now?"
        })

    # Constraint Extraction
    constraints = extract_constraints(msg)

    # State Persistence across queries
    if constraints["category"]:
        u_session["last_category"] = constraints["category"]
        session.modified = True
    elif constraints["location"] and not constraints["category"] and u_session.get("last_category"):
        constraints["category"] = u_session["last_category"]

    if constraints["location"]:
        u_session["location"] = constraints["location"]
        session.modified = True

    # Search Execution
    # Search Execution
    ranked_results = score_and_rank_places(constraints, u_session.get("location"))

    if ranked_results:
        u_session["last_places"] = [item["place"] for item in ranked_results[:3]]
        session.modified = True
        return jsonify({
            "reply": format_place_output(
                ranked_results, 
                display_name, 
                has_category=bool(constraints.get("category"))
            )
        })

    return jsonify({
        "reply": f"I couldn't find any places matching your request, {display_name}."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)