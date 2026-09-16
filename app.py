import json
import re
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = "roamie_full_nlp_pipeline_production_key"

STATE_AWAITING_NAME = "AWAITING_NAME"
STATE_IDLE = "IDLE"

MUNICIPALITIES = [
    "laoag", "batac", "san nicolas", "bacarra", "paoay", 
    "currimao", "dingras", "pagudpud", "bangui", "burgos", 
    "sarrat", "vintar", "piddig", "pasuquin", "solsona", "badoc"
]

CATEGORY_SYNONYMS = {
    "restaurant": ["restaurant", "bistro", "eatery", "dining", "grill", "steak", "steaks", "food", "wings", "pasta", "sushi", "ramen", "salmon", "seafood", "samgyupsal", "samgyup", "korean bbq", "bbq", "eat", "fastfood"],
    "cafe": ["cafe", "coffee", "bakeshop", "bakery", "espresso", "latte", "tea"],
    "karaoke": ["karaoke", "ktv", "singing", "music lounge"],
    "gym": ["gym", "fitness", "workout"],
    "hotel": ["hotel", "resort", "inn", "lodging", "stay"],
    "hospital": ["hospital", "medical center", "clinic", "health center", "infirmary"],
    "bank": ["bank", "atm"],
    "gas station": ["gas station", "gas", "fuel", "petrol"]
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
    "does", "do", "serve", "serving", "okay", "ok", "at", "here", "recommend", "me", "eat"
}

GREETINGS = {"hello", "hi", "hey", "sup", "good morning", "good afternoon", "good evening"}

CHITCHAT_RESPONSES = {
    "how are you": "I'm doing great! Ready to help you discover places in Ilocos Norte.",
    "who are you": "I'm Roamie, your local place finder for Ilocos Norte.",
    "what can you do": "I can help you find restaurants, cafes, hotels, hospitals, and spots matching your preferences.",
    "thank you": "You're welcome!",
    "thanks": "Anytime! Let me know if you need more recommendations.",
    "bye": "Goodbye! Have a great time exploring!"
}

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

def get_places_for_location(location_name):
    if not isinstance(dataset, dict):
        return get_all_places()
    
    for key, value in dataset.items():
        if key.strip().lower() == location_name.strip().lower():
            return value
    return []

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
            "last_places": []
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

def parse_business_hours(hours_str):
    if not hours_str or not isinstance(hours_str, str):
        return None, None
    matches = re.findall(r'(\d{1,2})(?::\d{2})?\s*(AM|PM|am|pm)', hours_str)
    if len(matches) >= 2:
        open_h = parse_time_to_24h(matches[0][0], matches[0][1])
        close_h = parse_time_to_24h(matches[1][0], matches[1][1])
        return open_h, close_h
    return None, None

def parse_price_range(price_str):
    if not price_str or not isinstance(price_str, str):
        return 0, 999999
    numbers = re.findall(r'\d+', price_str.replace(',', ''))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    elif len(numbers) == 1:
        return 0, int(numbers[0])
    return 0, 999999

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

    # Negation Handling
    negation_matches = re.findall(r'\b(?:no|not|without)\s+([a-z]+)\b', clean_text)
    constraints["negations"] = negation_matches

    # Location Extraction
    for muni in MUNICIPALITIES:
        if re.search(rf"\b{re.escape(muni)}\b", clean_text):
            constraints["location"] = muni
            break

    # Category Extraction
    for cat, syns in CATEGORY_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(s)}\b", clean_text) for s in syns):
            constraints["category"] = cat
            break

    # Time Expression Extraction
    time_match = re.search(r'(?:until|open until|at|till|closing at)\s*(\d{1,2})\s*(:\d{2})?\s*(am|pm)', clean_text)
    if time_match:
        constraints["closing_time_req"] = parse_time_to_24h(time_match.group(1), time_match.group(3))

    # Budget Extraction
    budget_match = re.search(r'(?:under|below|less than|max|budget of)\s*(?:₱|p)?\s*(\d+)', clean_text)
    if budget_match:
        constraints["max_budget"] = int(budget_match.group(1))

    # Strict Feature Extraction
    for token in tokens:
        if (token not in STOP_WORDS and 
            token not in MUNICIPALITIES and 
            token != constraints["category"] and 
            token not in constraints["negations"] and 
            len(token) >= 3):
            constraints["features"].append(token)

    return constraints

def score_and_rank_places(constraints, user_location):
    target_location = constraints["location"] or user_location
    scored_results = []

    if target_location:
        places_to_search = get_places_for_location(target_location)
    else:
        places_to_search = get_all_places()

    for place in places_to_search:
        score = 0
        match_reasons = []

        p_name = get_place_attr(place, ["Business name", "name", "business_name"], "")
        p_cat = str(get_place_attr(place, ["Category", "category"], "")).lower()
        p_city = str(get_place_attr(place, ["Municipality or city", "city", "municipality"], "")).lower()
        p_hours = get_place_attr(place, ["Opening and closing hours", "Operating Hours", "hours"], "")
        p_price = get_place_attr(place, ["Price range", "Price Range", "price_range"], "")
        
        p_features = [f.lower() for f in get_place_attr(place, ["Features", "features"], [])]
        p_keywords = [k.lower() for k in get_place_attr(place, ["Keywords and synonyms", "Keywords", "keywords"], [])]
        p_desc = str(get_place_attr(place, ["Short description", "Description", "description"], "")).lower()

        searchable_corpus = f"{p_name.lower()} {p_desc} {' '.join(p_features)} {' '.join(p_keywords)}"

        if any(neg in searchable_corpus for neg in constraints["negations"]):
            continue

        if target_location and target_location.lower() in p_city.lower():
            score += 40
            match_reasons.append(f"Located in {target_location.title()}")

        if constraints["category"]:
            cat_synonyms = CATEGORY_SYNONYMS.get(constraints["category"], [constraints["category"]])
            if any(syn in p_cat or syn in searchable_corpus for syn in cat_synonyms):
                score += 50
                match_reasons.append(f"Matches category '{constraints['category'].capitalize()}'")
            else:
                continue

        if constraints["closing_time_req"] is not None:
            open_h, close_h = parse_business_hours(p_hours)
            if close_h is not None and (close_h >= constraints["closing_time_req"] or close_h == 0):
                score += 20
                match_reasons.append("Open during requested hours")

        if constraints["max_budget"] is not None:
            p_min, p_max = parse_price_range(p_price)
            if p_min <= constraints["max_budget"]:
                score += 20
                match_reasons.append("Fits within budget")

        matched_feats = []
        for token in constraints["features"]:
            if re.search(rf"\b{re.escape(token)}\b", searchable_corpus):
                score += 25
                matched_feats.append(token)
        if matched_feats:
            match_reasons.append(f"Matches query term(s): {', '.join(matched_feats)}")

        if constraints["category"] and score >= 40:
            scored_results.append({"place": place, "score": score, "reasons": match_reasons})
        elif not constraints["category"] and score >= 50:
            scored_results.append({"place": place, "score": score, "reasons": match_reasons})

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:3]

def format_place_output(ranked_results, user_name):
    if not ranked_results:
        return f"I couldn't find any places matching all your preferences, {user_name}."

    reply = f"Here are the top options matching your query, {user_name}:<br><br>"
    for idx, item in enumerate(ranked_results, 1):
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

        if idx < len(ranked_results):
            reply += "<hr style='border: 0; border-top: 1px solid #444; margin: 10px 0;'>"

    reply += "<br><b>Which one suits you?</b> (Reply with 1, 2, or 3 to view description and full details!)"
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
        "last_places": []
    }
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    msg = data.get("message", "").strip()
    clean_text, tokens = normalize_and_tokenize(msg)
    msg_lower = clean_text
    
    u_session = get_user_session()

    # 1. Clean Name Capture Fix
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

    for pattern, response in CHITCHAT_RESPONSES.items():
        if pattern in msg_lower:
            return jsonify({"reply": response})

    # Extract constraints from message
    constraints = extract_constraints(msg)

    if constraints["location"] and not constraints["category"] and not constraints["features"] and constraints["closing_time_req"] is None and constraints["max_budget"] is None:
        u_session["location"] = constraints["location"]
        session.modified = True
        city_title = constraints["location"].title()
        return jsonify({
            "reply": f"Got it, you're looking in <b>{city_title}</b>! What specific place or category are you looking for? (e.g., restaurant, cafe, bank, gas station, hotel)"
        })

    target_place = None
    all_places = get_all_places()
    for place in all_places:
        p_name = get_place_attr(place, ["Business name", "name", "business_name"], "").lower()
        if p_name and p_name in msg_lower:
            target_place = place
            break

    is_followup_question = any(q in msg_lower for q in ["do they", "does it", "is there", "has", "have", "with", "serve", "serving"])

    if target_place and not is_followup_question:
        return jsonify({"reply": format_single_place_detail(target_place, display_name)})

    if is_followup_question or (target_place and is_followup_question):
        query_words = [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]
        if target_place:
            p_words = normalize_and_tokenize(get_place_attr(target_place, ["Business name", "name"], ""))[1]
            query_words = [w for w in query_words if w not in p_words]

        feature_to_check = query_words[0] if query_words else None
        places_to_inspect = [target_place] if target_place else u_session["last_places"]

        if places_to_inspect and feature_to_check:
            reply = f"Here is what I found regarding <b>{feature_to_check}</b>, {display_name}:<br><br>"
            for place in places_to_inspect:
                p_name = get_place_attr(place, ["Business name", "name", "business_name"])
                p_feats = [f.lower() for f in get_place_attr(place, ["Features", "features"], [])]
                p_keywords = [k.lower() for k in get_place_attr(place, ["Keywords and synonyms", "Keywords", "keywords"], [])]
                p_desc = str(get_place_attr(place, ["Short description", "Description", "description"], "")).lower()
                full_text = f"{p_desc} {' '.join(p_feats)} {' '.join(p_keywords)}"

                if re.search(rf"\b{re.escape(feature_to_check)}\b", full_text):
                    reply += f"✅ <b>{p_name}</b>: Yes, they offer/have {feature_to_check}.<br>"
                else:
                    reply += f"❌ <b>{p_name}</b>: No explicit mention of {feature_to_check}.<br>"

            return jsonify({"reply": reply.strip()})

    # 5. Standard NLP Search Pipeline
    if constraints["location"]:
        u_session["location"] = constraints["location"]
        session.modified = True

    search_verbs = ["find", "search", "where", "looking", "recommend", "show", "list", "want", "place", "spots", "hospital", "salmon", "steak", "bank"]
    has_search_intent = (
        constraints["category"] is not None or 
        constraints["location"] is not None or 
        constraints["closing_time_req"] is not None or 
        constraints["max_budget"] is not None or 
        len(constraints["features"]) > 0 or
        any(v in msg_lower for v in search_verbs)
    )

    if has_search_intent:
        ranked_results = score_and_rank_places(constraints, u_session["location"])
        if ranked_results:
            u_session["last_places"] = [item["place"] for item in ranked_results]
            session.modified = True
            return jsonify({"reply": format_place_output(ranked_results, display_name)})

    # 6. Fallback
    return jsonify({
        "reply": f"I couldn't find any places matching '{msg}', {display_name}."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)