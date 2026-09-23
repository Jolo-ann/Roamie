import json
import re
import random
import datetime
import nltk
import spacy
from flask import Flask, request, jsonify, render_template, session
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

#  -recommended nltk.org nlp modules
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
    "restaurant": ["restaurant", "bistro", "eatery", "dining", "grill", "steak", "food", "wings", "pasta", "sushi", "ramen", "seafood", "samgyupsal", "samgyup", "bbq", "eat", "fastfood"],
    "cafe": ["cafe", "coffee", "bakeshop", "bakery", "espresso", "latte", "tea"],
    "karaoke": ["karaoke", "ktv", "singing", "music lounge"],
    "gym": ["gym", "fitness", "workout"],
    "hotel": ["hotel", "resort", "inn", "lodging", "stay"],
    "hospital": ["hospital", "medical center", "clinic", "health center", "infirmary"],
    "bank": ["bank", "atm"],
    "gas station": ["gas station", "gasoline station", "gasoline", "gas", "fuel", "petrol"]
}

FEATURE_ALIASES = {
    "parking": ["parking", "parking space", "car park", "parking lot"],
    "wifi": ["wifi", "wi-fi", "internet", "free wifi"],
    "ktv": ["ktv", "karaoke", "singing room"],
    "pet-friendly": ["pet", "pets", "pet-friendly", "dog friendly", "cat friendly"],
    "aircon": ["aircon", "air conditioning", "ac", "air conditioned"],
    "outdoor": ["outdoor", "al fresco", "outdoor seating"],
    "card": ["card", "credit card", "debit card", "gcash"],
    "delivery": ["delivery", "takeout", "takeaway"]
}

FEATURE_KEYWORDS = list(FEATURE_ALIASES.keys())

THINKING_PHRASES = ["still thinking", "wait", "hold on", "give me a sec", "give me a minute", "thinking", "not sure yet", "wait a bit"]
READY_PHRASES = ["im ready", "i'm ready", "ready", "ok ready", "okay ready"]
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

ALL_PLACES = get_all_places()

def build_place_corpus(place):
    name = str(place.get("Business name") or place.get("name") or "")
    cat = str(place.get("Category") or place.get("category") or "")
    city = str(place.get("Municipality or city") or place.get("city") or "")
    desc = str(place.get("Short description") or place.get("Description") or "")
    
    feats = place.get("Features", [])
    if isinstance(feats, list):
        feats_str = " ".join([str(f) for f in feats])
    else:
        feats_str = str(feats)

    keywords = str(place.get("Keywords and synonyms") or "")
    return f"{name}. Category: {cat}. Location: {city}. Details: {desc} Features: {feats_str} Keywords: {keywords}"

PLACE_CORPUS = [build_place_corpus(p) for p in ALL_PLACES]
PLACE_EMBEDDINGS = embedder.encode(PLACE_CORPUS, convert_to_tensor=True) if PLACE_CORPUS else None

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

def find_place_by_name(query_text):
    """Searches ALL_PLACES strictly when a specific business name is requested."""
    query_clean = re.sub(r'[^a-z0-9\s]', '', query_text.lower()).strip()
    
    category_words = [w for syns in CATEGORY_SYNONYMS.values() for w in syns]
    if any(re.search(rf"\b{re.escape(cw)}\b", query_clean) for cw in category_words) and not any(kw in query_clean for kw in ["does", "is there", "has", "have"]):
        return None

    for place in ALL_PLACES:
        p_name = str(get_place_attr(place, ["Business name", "name", "business_name"])).lower()
        p_name_clean = re.sub(r'[^a-z0-9\s]', '', p_name).strip()
        
        if p_name_clean and p_name_clean in query_clean:
            return place
            
        tokens = [t for t in p_name_clean.split() if len(t) > 2 and t not in ["gas", "station", "the", "and", "hall", "center", "store"]]
        if len(tokens) >= 2 and " ".join(tokens) in query_clean:
            return place

    return None

def parse_time_to_minutes(time_str):
    time_str = time_str.strip().upper()
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?', time_str)
    if not match:
        return None
        
    hours = int(match.group(1))
    minutes = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    if period == "PM" and hours != 12:
        hours += 12
    elif period == "AM" and hours == 12:
        hours = 0
    elif period is None and hours < 7:
        hours += 12
        
    return hours * 60 + minutes

def parse_hours_range(hours_str):
    try:
        parts = re.split(r'–|-|to', hours_str, flags=re.IGNORECASE)
        if len(parts) != 2:
            return None, None
            
        open_min = parse_time_to_minutes(parts[0])
        close_min = parse_time_to_minutes(parts[1])
        
        if close_min is not None and open_min is not None and close_min <= open_min:
            close_min += 24 * 60
            
        return open_min, close_min
    except Exception:
        return None, None

def is_explicit_followup_question(msg_lower):
    followup_patterns = [
        r"\bdo they\b", r"\bis there\b", r"\bdoes it\b", r"\bare there\b", 
        r"\bhas parking\b", r"\bhave parking\b", r"\bhas wifi\b", r"\bhave wifi\b",
        r"\bhas ktv\b", r"\bhave ktv\b", r"\bwhich one\b"
    ]
    return any(re.search(pat, msg_lower) for pat in followup_patterns)

def handle_followup_feature_question(msg_lower, last_places, user_name):
    if not is_explicit_followup_question(msg_lower):
        return None

    detected_feature = None
    for feat, aliases in FEATURE_ALIASES.items():
        if any(alias in msg_lower for alias in aliases):
            detected_feature = feat
            break

    if not detected_feature:
        return None

    reply = f"Here is the info regarding <b>{detected_feature.title()}</b> for the places listed above, {user_name}:<br><br>"
    
    for idx, place in enumerate(last_places, 1):
        name = get_place_attr(place, ["Business name", "name", "business_name"])
        
        feats_raw = get_place_attr(place, ["Features", "features"], [])
        if isinstance(feats_raw, list):
            p_feats = [str(f).lower() for f in feats_raw]
        else:
            p_feats = [str(feats_raw).lower()]
            
        p_desc = str(get_place_attr(place, ["Short description", "Description", "description"], "")).lower()
        p_kw = str(get_place_attr(place, ["Keywords and synonyms", "keywords"], "")).lower()
        
        combined_info = f"{' '.join(p_feats)} {p_desc} {p_kw}"
        aliases = FEATURE_ALIASES.get(detected_feature, [detected_feature])

        if any(alias in combined_info for alias in aliases):
            reply += f"✅ <b>{idx}. {name}</b>: Yes, feature/info found matching <i>{detected_feature}</i>.<br>"
        else:
            reply += f"❌ <b>{idx}. {name}</b>: No explicit mention of <i>{detected_feature}</i> in details.<br>"

    reply += "<br>Reply with 1, 2, or 3 to view full details for any option!"
    return reply

def extract_constraints_nlp(query_text):
    doc = nlp(query_text)
    clean_text = query_text.lower()
    
    constraints = {
        "category": None,
        "location": None,
        "max_budget": None,
        "required_features": [],
        "target_closing_time": None,
        "raw_query": query_text
    }

    # 1. Location
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            ent_text = ent.text.lower()
            for muni in MUNICIPALITIES:
                if muni in ent_text:
                    constraints["location"] = muni
                    break

    if not constraints["location"]:
        for muni in MUNICIPALITIES:
            if re.search(rf"\b{re.escape(muni)}\b", clean_text):
                constraints["location"] = muni
                break

    # 2. Category
    lemmatized_tokens = [lemmatizer.lemmatize(token.text.lower()) for token in doc]
    lemmatized_text = " ".join(lemmatized_tokens)

    for cat, syns in CATEGORY_SYNONYMS.items():
        for syn in syns:
            syn_lemma = lemmatizer.lemmatize(syn)
            if re.search(rf"\b{re.escape(syn_lemma)}\b", lemmatized_text):
                constraints["category"] = cat
                break
        if constraints["category"]:
            break

    # 3. Features
    for feat, aliases in FEATURE_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", clean_text) for alias in aliases):
            constraints["required_features"].append(feat)

    # 4. Closing Time
    time_match = re.search(r'(?:until|till|open until|open till|past|at least)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', clean_text, re.IGNORECASE)
    if time_match:
        raw_time = time_match.group(1).strip()
        constraints["target_closing_time"] = parse_time_to_minutes(raw_time)

    # 5. Budget
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CARDINAL"]:
            digits = re.sub(r'[^\d]', '', ent.text)
            if digits and int(digits) > 10:
                constraints["max_budget"] = int(digits)

    return constraints

# Scoring ^ Ranking
def score_and_rank_places_nlp(constraints, session_location=None):
    if not ALL_PLACES:
        return []

    req_category = constraints.get("category")
    req_location = constraints.get("location") or session_location
    req_features = constraints.get("required_features", [])
    target_close_min = constraints.get("target_closing_time")
    req_budget = constraints.get("max_budget")
    query_text = constraints.get("raw_query", "")

    query_embedding = embedder.encode(query_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, PLACE_EMBEDDINGS)[0]

    category_synonyms = CATEGORY_SYNONYMS.get(req_category, [req_category]) if req_category else []
    scored_places = []

    for idx, place in enumerate(ALL_PLACES):
        p_cat = str(get_place_attr(place, ["Category", "category"], "")).lower()
        p_city = str(get_place_attr(place, ["Municipality or city", "city", "municipality"], "")).lower()
        p_hours = str(get_place_attr(place, ["Opening and closing hours", "Operating Hours", "hours"], ""))
        p_price = str(get_place_attr(place, ["Price range", "Price Range", "price_range"], ""))
        
        full_text = PLACE_CORPUS[idx].lower()

        # Hard Filter 1: Location
        if req_location:
            clean_req = req_location.lower().replace("city", "").strip()
            clean_p = p_city.lower().replace("city", "").strip()
            if clean_req not in clean_p:
                continue

        # Hard Filter 2: Category
        if req_category:
            is_cat_match = any(re.search(rf"\b{re.escape(syn)}\b", full_text) for syn in category_synonyms)
            if not is_cat_match:
                continue

        # Base Score (Up to 50 pts)
        score = float(cosine_scores[idx]) * 50

        # Time Bonus / Penalty
        if target_close_min is not None and p_hours != "N/A":
            open_min, close_min = parse_hours_range(p_hours)
            if close_min is not None:
                if close_min >= target_close_min:
                    score += 25.0
                else:
                    score -= 20.0

        # Feature Match Bonus
        if req_features:
            feats_raw = get_place_attr(place, ["Features", "features"], [])
            p_feats = [str(f).lower() for f in feats_raw] if isinstance(feats_raw, list) else [str(feats_raw).lower()]
            p_desc = str(get_place_attr(place, ["Short description", "Description", "description"], "")).lower()
            p_kw = str(get_place_attr(place, ["Keywords and synonyms", "keywords"], "")).lower()
            
            combined_feat_text = f"{' '.join(p_feats)} {p_desc} {p_kw} {full_text}"

            matched_count = 0
            for feat in req_features:
                aliases = FEATURE_ALIASES.get(feat, [feat])
                if any(alias in combined_feat_text for alias in aliases):
                    matched_count += 1
            
            score += (matched_count / len(req_features)) * 25.0

        scored_places.append({
            "place": place, 
            "score": round(score, 2)
        })

    scored_places.sort(key=lambda x: x["score"], reverse=True)
    return scored_places

def format_place_output(ranked_results, user_name, has_category=True):
    top_results = ranked_results[:3]
    if not top_results:
        return f"I couldn't find any places matching all your criteria, {user_name}."

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
    features = ", ".join(features_list) if isinstance(features_list, list) else str(features_list)
    description = get_place_attr(place, ["Short description", "Description", "description"], default="No description available.")

    # Google Maps Link -check dataset url for maps
    maps_url = get_place_attr(place, ["Google Maps", "maps_link", "GoogleMaps", "maps"], default=None)
    if not maps_url or maps_url == "N/A":
        search_query = f"{name} {city}".replace(" ", "+")
        maps_url = f"https://www.google.com/maps/search/?api=1&query={search_query}"

    reply = f"Here are the full details for <b>{name}</b>, {user_name}:<br><br>"
    reply += f"🏢 <b>Category</b>: {cat}<br>"
    reply += f"📍 <b>Location</b>: {city}<br>"
    reply += f"⏰ <b>Hours</b>: {hours}<br>"
    reply += f"💰 <b>Price Range</b>: {price}<br>"
    if features and features != "N/A": 
        reply += f"✨ <b>Features</b>: {features}<br>"
        
    reply += f"🗺️ <b>Google Maps</b>: <a href='{maps_url}' target='_blank' style='color: #4da6ff; text-decoration: underline;'>Open in Google Maps ↗</a><br>"
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
    msg_lower = msg.lower()
    
    u_session = get_user_session()

    # Name Capture Phase
    if u_session["state"] == STATE_AWAITING_NAME or u_session["name"] is None:
        clean_name = re.sub(r'^(hello|hi|hey)?\s*,?\s*(my name is|call me|im|i\'m|i am)\s*', '', msg, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r'^(hello|hi|hey)\s+', '', clean_name, flags=re.IGNORECASE).strip()
        
        u_session["name"] = clean_name.title() if clean_name else "Friend"
        u_session["state"] = STATE_IDLE
        session.modified = True
        return jsonify({
            "reply": f"Nice to meet you, {u_session['name']}! What kind of place are you looking for in Ilocos Norte today?"
        })

    display_name = u_session["name"]

    # DIRECT SPECIFIC VENUE LOOKUP
    target_place = find_place_by_name(msg)
    detected_feature = None
    for feat, aliases in FEATURE_ALIASES.items():
        if any(alias in msg_lower for alias in aliases):
            detected_feature = feat
            break

    if target_place and detected_feature:
        p_name = get_place_attr(target_place, ["Business name", "name", "business_name"])
        feats_raw = get_place_attr(target_place, ["Features", "features"], [])
        p_feats = [str(f).lower() for f in feats_raw] if isinstance(feats_raw, list) else [str(feats_raw).lower()]
        p_desc = str(get_place_attr(target_place, ["Short description", "Description"], "")).lower()
        p_kw = str(get_place_attr(target_place, ["Keywords and synonyms", "keywords"], "")).lower()
        
        combined_info = f"{' '.join(p_feats)} {p_desc} {p_kw}"
        aliases = FEATURE_ALIASES.get(detected_feature, [detected_feature])

        if any(alias in combined_info for alias in aliases):
            reply = f"✅ Yes, <b>{p_name}</b> has <b>{detected_feature.title()}</b>, {display_name}!"
        else:
            reply = f"❌ No explicit mention of <b>{detected_feature.title()}</b> for <b>{p_name}</b> in our records, {display_name}."
            
        return jsonify({"reply": reply})

    # Selection Phase (1, 2, 3) or more maybe 5
    if u_session["last_places"] and (msg_lower in ["1", "2", "3", "option 1", "option 2", "option 3", "first", "second", "third"]):
        idx_map = {"1": 0, "option 1": 0, "first": 0, "2": 1, "option 2": 1, "second": 1, "3": 2, "option 3": 2, "third": 2}
        selected_idx = idx_map.get(msg_lower)
        if selected_idx is not None and selected_idx < len(u_session["last_places"]):
            chosen_place = u_session["last_places"][selected_idx]
            return jsonify({"reply": format_single_place_detail(chosen_place, display_name)})

    # Followup Feature Interception on Previous Active Results
    if u_session["last_places"]:
        followup_reply = handle_followup_feature_question(msg_lower, u_session["last_places"], display_name)
        if followup_reply:
            return jsonify({"reply": followup_reply})

    if msg_lower in ["okay", "ok"]:
        return jsonify({"reply": f"Got it, {display_name}! What place or category would you like to check out next?"})

    if any(phrase in msg_lower for phrase in THINKING_PHRASES):
        return jsonify({"reply": f"No rush at all, {display_name}! Take your time. Just let me know whenever you're ready to search."})

    if any(phrase in msg_lower for phrase in READY_PHRASES):
        return jsonify({"reply": f"Awesome! What kind of place or category would you like to search for, {display_name}?"})

    if msg_lower in GREETINGS:
        return jsonify({"reply": f"Hello {display_name}! How can I help you explore Ilocos Norte right now?"})

    # Constraint Extraction
    constraints = extract_constraints_nlp(msg)

    # Session State Persistence
    has_explicit_category = bool(constraints["category"])
    
    if constraints["category"]:
        u_session["last_category"] = constraints["category"]
        session.modified = True
    elif constraints["location"] and not constraints["category"] and u_session.get("last_category"):
        constraints["category"] = u_session["last_category"]
        has_explicit_category = True

    if constraints["location"]:
        u_session["location"] = constraints["location"]
        session.modified = True

    # Search Execution
    ranked_results = score_and_rank_places_nlp(constraints, u_session.get("location"))

    if ranked_results:
        u_session["last_places"] = [item["place"] for item in ranked_results[:3]]
        session.modified = True
        return jsonify({
            "reply": format_place_output(
                ranked_results, 
                display_name, 
                has_category=has_explicit_category
            )
        })

    return jsonify({"reply": f"I couldn't find any places matching your request, {display_name}."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)