import json
import re
import random
import datetime
import math
from collections import Counter
from difflib import SequenceMatcher

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.tree import Tree
from flask import Flask, request, jsonify, render_template, session

# 1. APPLICATION & SYSTEM STATE INITIALIZATION
# Flask server setup
app = Flask(__name__)
app.secret_key = "roamie_production_key_fixed"

STATE_AWAITING_NAME = "AWAITING_NAME"
STATE_IDLE = "IDLE"

# 2. LOCAL ENTITIES GAZETTEER & SCOPE DEFINITIONS
MUNICIPALITIES = ["laoag", "batac", "san nicolas", "bacarra", "paoay"]

MUNICIPALITY_ALIASES = {
    "laoag": ["laoag", "laoag city"],
    "batac": ["batac", "batac city"],
    "san nicolas": ["san nicolas"],
    "bacarra": ["bacarra"],
    "paoay": ["paoay"],
}

# Explicitly track nearby Ilocos Norte towns that are OUT OF SCOPE
# so the engine can catch them and provide a helpful fallback response.
UNSUPPORTED_LOCATIONS = [
    "pagudpud", "bangui", "burgos", "sarrat", "vintar", 
    "piddig", "pasuquin", "solsona", "badoc", "currimao", "dingras"
]

CATEGORY_SYNONYMS = {
    "restaurant": [
        "restaurant", "restaurants", "resto", "bistro", "eatery",
        "dining", "diner", "grill", "steak", "food", "foods",
        "wings", "pasta", "sushi", "ramen", "seafood",
        "samgyupsal", "samgyup", "bbq", "barbecue", "eat",
        "fastfood", "fast food", "food place", "empanada", "empanadaan",
        "bagnet", "poqui-poqui", "poquipoqui", "poqui poqui", "ilocano food", "local dishes"
    ],
    "cafe": [
        "cafe", "cafes", "coffee", "coffee shop", "bakeshop",
        "bakery", "espresso", "latte", "tea", "tea shop", "cozy cafe", "cozy café", "café"
    ],
    "karaoke": [
        "karaoke", "ktv", "singing", "music lounge", "videoke"
    ],
    "gym": [
        "gym", "fitness", "workout", "fitness center"
    ],
    "hotel": [
        "hotel", "hotels", "resort", "resorts", "inn",
        "lodging", "stay", "accommodation", "place to stay"
    ],
    "hospital": [
        "hospital", "medical center", "clinic", "health center",
        "infirmary", "healthcare"
    ],
    "bank": [
        "bank", "banks", "atm", "banking"
    ],
    "gas station": [
        "gas station", "gasoline station", "gasoline",
        "gas", "fuel", "petrol", "petrol station"
    ],
    "supermarket": [
        "supermarket", "supermarkets", "grocery", "grocery store", "groceries",
        "mart", "convenience store", "hypermarket", "department store"
    ]
}

LOCAL_DISHES = [
    "empanada", "bagnet", "poqui poqui", "poqui-poqui", "poquipoqui",
    "longganisa", "igado", "dinakdakan", "pinakbet", "royal bibingka", "miki"
]

FEATURE_ALIASES = {
    "parking": ["parking", "parking space", "car park", "parking lot", "with parking", "has parking"],
    "wifi": ["wifi", "wi-fi", "internet", "free wifi", "free wi-fi", "internet access", "good wifi", "good wi-fi", "working"],
    "ktv": ["ktv", "karaoke", "singing room", "videoke"],
    "pet-friendly": ["pet", "pets", "pet-friendly", "pet friendly", "dog friendly", "cat friendly"],
    "aircon": ["aircon", "air conditioning", "air conditioned", "air-conditioned", "ac"],
    "outdoor": ["outdoor", "al fresco", "outdoor seating", "outside seating"],
    "card": ["card", "credit card", "debit card", "gcash", "cashless", "card payment"],
    "delivery": ["delivery", "deliver", "takeout", "takeaway", "food delivery"],
}

FEATURE_KEYWORDS = list(FEATURE_ALIASES.keys())
THINKING_PHRASES = ["still thinking", "wait", "hold on", "give me a sec", "give me a minute", "thinking", "not sure yet", "wait a bit"]
READY_PHRASES = ["im ready", "i'm ready", "ready", "ok ready", "okay ready"]
GREETINGS = {"hello", "hi", "hey", "sup", "good morning", "good afternoon", "good evening"}


# 3. DATASET LOADING & CORPUS CONSTRUCTION
def load_dataset():
    """Loads the database from JSON."""
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset.json: {e}")
        return {}

dataset = load_dataset()

def get_all_places():
    """Flattens dataset into a unified list of place records."""
    if isinstance(dataset, dict):
        places = []
        for city_places in dataset.values():
            if isinstance(city_places, list):
                places.extend(city_places)
        return places
    return dataset if isinstance(dataset, list) else []

ALL_PLACES = get_all_places()

def build_place_corpus(place):
    name = str(place.get("Business name") or place.get("business_name") or place.get("name") or "")
    cat = str(place.get("Category") or place.get("category") or "")
    city = str(place.get("Municipality or city") or place.get("municipality_city") or place.get("city") or "")
    desc = str(place.get("Short description") or place.get("short_description") or place.get("Description") or "")
    feats = place.get("Features") or place.get("features") or []
    feats_str = " ".join(str(f) for f in feats) if isinstance(feats, list) else str(feats)
    keywords = str(place.get("Keywords and synonyms") or place.get("keywords_synonyms") or place.get("keywords") or "")

    return f"{name}. Category: {cat}. Location: {city}. Details: {desc} Features: {feats_str} Keywords: {keywords}"

PLACE_CORPUS = [build_place_corpus(p) for p in ALL_PLACES]

def get_place_attr(place, keys, default="N/A"):
    for key in keys:
        if key in place and place[key] not in [None, "", "N/A"]:
            return place[key]
    return default

# 4. TEXT PROCESSING & TOKENIZATION UTILITIES
def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("’", "'").replace('"', '')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_text(text):
    """Uses NLTK's wordpunct_tokenize for token boundaries."""
    return wordpunct_tokenize(normalize_text(text))

def singularize_word(word):
    word = word.lower().strip()
    irregular = {
        "restaurants": "restaurant", "cafes": "cafe", "hotels": "hotel",
        "hospitals": "hospital", "banks": "bank", "gyms": "gym",
        "resorts": "resort", "stores": "store", "places": "place",
        "groceries": "grocery", "supermarkets": "supermarket"
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def normalized_tokens(text):
    return [
        singularize_word(token)
        for token in tokenize_text(text)
        if re.search(r"[a-z0-9]", token)
    ]

# 5. FUZZY MATCHING & GAZETTEER ENTITY RECOGNITION
def similarity_score(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def fuzzy_phrase_present(text, phrase, threshold=0.80):
    clean_text = normalize_text(text)
    clean_phrase = normalize_text(phrase)
    
    if not clean_phrase or not clean_text:
        return False
        
    if re.search(rf"(?<!\w){re.escape(clean_phrase)}(?!\w)", clean_text):
        return True
        
    text_words = clean_text.split()
    phrase_words = clean_phrase.split()
    phrase_len = len(phrase_words)
    
    for i in range(len(text_words) - phrase_len + 1):
        window = " ".join(text_words[i:i + phrase_len])
        if similarity_score(window, clean_phrase) >= threshold:
            return True
            
    return False

def phrase_present(text, phrase):
    return fuzzy_phrase_present(text, phrase, threshold=0.82)

def detect_location_entities(text):
    clean = normalize_text(text)
    found_locations = set()
    for municipality, aliases in MUNICIPALITY_ALIASES.items():
        for alias in aliases:
            if phrase_present(clean, alias):
                found_locations.add(municipality)
                break
    return list(found_locations)

def detect_unsupported_location(text):
    clean = normalize_text(text)
    for loc in UNSUPPORTED_LOCATIONS:
        if phrase_present(clean, loc):
            return loc.title()
    return None

def detect_category_entity(text):
    clean = normalize_text(text)
    matches = []
    for category, aliases in CATEGORY_SYNONYMS.items():
        for alias in aliases:
            if phrase_present(clean, alias):
                matches.append((len(alias), category))
                break
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]

def detect_feature_entities(text):
    clean = normalize_text(text)
    found = []
    for feature, aliases in FEATURE_ALIASES.items():
        if any(phrase_present(clean, alias) for alias in aliases):
            found.append(feature)
    return found

def detect_dish_entity(text):
    clean = normalize_text(text)
    for dish in LOCAL_DISHES:
        if phrase_present(clean, dish):
            return dish
    return None

# 6. REGEX EXTRACTION
def parse_time_to_minutes(time_str):
    time_str = str(time_str or "").strip().upper()
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?", time_str)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    if hours > 23 or minutes > 59:
        return None

    if period == "PM" and hours != 12:
        hours += 12
    elif period == "AM" and hours == 12:
        hours = 0
    elif period is None and hours < 7:
        hours += 12

    return hours * 60 + minutes

def parse_hours_range(hours_str):
    try:
        clean = str(hours_str or "").strip()
        parts = re.split(r"\s*(?:-|–|to)\s*", clean, flags=re.IGNORECASE)
        if len(parts) != 2:
            return None, None

        open_min = parse_time_to_minutes(parts[0])
        close_min = parse_time_to_minutes(parts[1])

        if close_min is not None and open_min is not None and close_min <= open_min:
            close_min += 24 * 60  # Overnight business shift handling

        return open_min, close_min
    except Exception:
        return None, None

def parse_budget(query_text):
    clean = normalize_text(query_text)
    patterns = [
        r"(?:₱|php\s*|p\s*)\s*(\d[\d,]*(?:\.\d+)?)",
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:pesos?|php)",
        r"\b(?:under|below|less than|up to|within|max(?:imum)?(?: budget)? of|cheap|affordable|budget)\b.*?(?:₱|php|p)?\s*(\d[\d,]*)",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            try:
                raw = match.group(1).replace(",", "")
                value = float(raw)
                if value > 10:
                    candidates.append(int(value))
            except (ValueError, IndexError):
                pass

    budget_words = ["budget", "price", "cost", "cheap", "affordable", "under", "below", "less than", "pesos", "php", "₱"]
    if not any(word in clean for word in budget_words):
        return None

    return min(candidates) if candidates else None

def parse_target_closing_time(query_text):
    clean = normalize_text(query_text)
    pattern = r"\b(?:until|till|open until|open till|past|at least|up to|open late|late night)\b\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
    match = re.search(pattern, clean, flags=re.IGNORECASE)
    if not match:
        return None
    return parse_time_to_minutes(match.group(1))

# 7. NATURAL LANGUAGE PARSER & QUERY TREE GENERATOR
def build_query_parse_tree(query_text, constraints):
    children = []
    if constraints.get("category"):
        children.append(Tree("CATEGORY", [constraints["category"]]))
    if constraints.get("locations"):
        children.append(Tree("LOCATIONS", constraints["locations"]))
    if constraints.get("required_features"):
        children.append(Tree("FEATURES", [Tree("FEATURE", [feature]) for feature in constraints["required_features"]]))
    if constraints.get("dish"):
        children.append(Tree("DISH", [constraints["dish"]]))
    if constraints.get("max_budget") is not None:
        children.append(Tree("BUDGET", [str(constraints["max_budget"])]))
    if constraints.get("target_closing_time") is not None:
        children.append(Tree("CLOSING_TIME", [str(constraints["target_closing_time"])]))
    if not children:
        children.append(Tree("TEXT", list(normalized_tokens(query_text))))
    return Tree("QUERY", children)

def extract_constraints_nlp(query_text):
    clean_text = normalize_text(query_text)
    tokens = tokenize_text(clean_text)
    locations = detect_location_entities(clean_text)

    constraints = {
        "category": detect_category_entity(clean_text),
        "locations": locations,
        "location": locations[0] if locations else None,
        "dish": detect_dish_entity(clean_text),
        "max_budget": parse_budget(clean_text),
        "required_features": detect_feature_entities(clean_text),
        "target_closing_time": parse_target_closing_time(clean_text),
        "raw_query": query_text,
        "tokens": tokens,
        "parse_tree": None,
    }

    constraints["parse_tree"] = build_query_parse_tree(query_text, constraints)
    return constraints

# 8. PURE PYTHON TF-IDF & LEXICAL SCORING ENGINE
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "i", "me",
    "my", "to", "for", "in", "at", "of", "and", "or",
    "with", "that", "this", "there", "any", "some",
    "place", "places", "looking", "find", "show",
    "give", "want", "need", "near", "around"
}

def content_tokens(text):
    return [
        singularize_word(token)
        for token in normalized_tokens(text)
        if token not in STOPWORDS and len(token) > 1
    ]

def lexical_overlap_score(query_text, place_text):
    q_tokens = content_tokens(query_text)
    p_tokens = content_tokens(place_text)

    if not q_tokens or not p_tokens:
        return 0.0

    num_docs = len(PLACE_CORPUS) if PLACE_CORPUS else 1
    doc_words = p_tokens
    word_counts = Counter(doc_words)

    tfidf_score = 0.0
    for token in set(q_tokens):
        matched_word = token if token in word_counts else None
        if not matched_word:
            for w in word_counts.keys():
                if similarity_score(token, w) >= 0.85:
                    matched_word = w
                    break
        if matched_word:
            tf = word_counts[matched_word] / len(doc_words)
            df = sum(1 for doc in PLACE_CORPUS if token in doc.lower())
            idf = math.log((1 + num_docs) / (1 + df)) + 1.0
            tfidf_score += tf * idf

    sequence = SequenceMatcher(None, normalize_text(query_text), normalize_text(place_text)).ratio()
    return (tfidf_score * 0.70) + (sequence * 0.30)

# 9. ATTRIBUTE MATCHING & SCORING RANKER
def category_matches(place, requested_category):
    if not requested_category:
        return True
    category_text = normalize_text(get_place_attr(place, ["Category", "category"], ""))
    aliases = CATEGORY_SYNONYMS.get(requested_category, [requested_category])
    return any(phrase_present(category_text, alias) for alias in aliases)

def location_matches(place, requested_locations):
    if not requested_locations:
        return True
    if isinstance(requested_locations, str):
        requested_locations = [requested_locations]
        
    city_text = normalize_text(get_place_attr(place, ["Municipality or city", "municipality_city", "city", "municipality"], ""))
    
    for req_loc in requested_locations:
        aliases = MUNICIPALITY_ALIASES.get(req_loc, [req_loc])
        if any(phrase_present(city_text, alias) for alias in aliases):
            return True
    return False

def place_has_feature(place, feature):
    feats_raw = get_place_attr(place, ["Features", "features"], [])
    feature_text = " ".join(str(item) for item in feats_raw) if isinstance(feats_raw, list) else str(feats_raw)
    description = str(get_place_attr(place, ["Short description", "short_description", "Description", "description"], ""))
    keywords = str(get_place_attr(place, ["Keywords and synonyms", "keywords_synonyms", "keywords"], ""))

    combined = normalize_text(f"{feature_text} {description} {keywords}")
    aliases = FEATURE_ALIASES.get(feature, [feature])
    return any(phrase_present(combined, alias) for alias in aliases)

def price_number_from_place(place):
    price = str(get_place_attr(place, ["Price range", "price_range", "Price Range"], ""))
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", price)
    values = []

    for number in numbers:
        try:
            values.append(float(number.replace(",", "")))
        except ValueError:
            pass

    return max(values) if values else None

def score_and_rank_places_nlp(constraints, session_location=None, ignore_features=False):
    if not ALL_PLACES:
        return []

    requested_category = constraints.get("category")
    requested_locations = constraints.get("locations") or ([session_location] if session_location else [])
    requested_features = constraints.get("required_features", []) if not ignore_features else []
    requested_dish = constraints.get("dish")
    target_close_min = constraints.get("target_closing_time")
    requested_budget = constraints.get("max_budget")
    query_text = constraints.get("raw_query", "")

    scored_places = []

    for place in ALL_PLACES:
        # STRICT LOCATION FILTERING: Hard discard venues not matching requested town
        if requested_locations and not location_matches(place, requested_locations):
            continue

        if requested_category and not category_matches(place, requested_category):
            continue

        place_corpus = build_place_corpus(place)
        score = 0.0

        # Lexical score weighting
        lexical_score = lexical_overlap_score(query_text, place_corpus)
        score += lexical_score * 35.0

        if requested_category:
            score += 25.0
        else:
            score += min(10.0, lexical_score * 10.0)

        if requested_locations:
            score += 15.0

        if requested_features:
            matched_count = sum(1 for feature in requested_features if place_has_feature(place, feature))
            if matched_count == 0 and not ignore_features:
                continue
            feature_ratio = matched_count / len(requested_features)
            score += feature_ratio * 20.0

        # Local Dish Scoring Logic (Boosts authentic local dining over chains)
        if requested_dish:
            p_keywords = str(get_place_attr(place, ["Keywords and synonyms", "keywords_synonyms", "keywords"], "")).lower()
            p_name = str(get_place_attr(place, ["Business name", "business_name", "name"], "")).lower()
            p_desc = str(get_place_attr(place, ["Short description", "short_description", "description"], "")).lower()
            p_cat = str(get_place_attr(place, ["Category", "category"], "")).lower()

            combined_place_text = f"{p_name} {p_keywords} {p_desc}"

            if phrase_present(combined_place_text, requested_dish):
                score += 50.0
            else:
                if any(chain in p_name or chain in p_cat for chain in ["mcdonald", "chowking", "jollibee", "mang inasal", "fastfood"]):
                    score -= 40.0
                else:
                    score -= 15.0

        if requested_budget is not None:
            place_price = price_number_from_place(place)
            if place_price is None:
                score += 2.0
            elif place_price <= requested_budget:
                score += 20.0
            elif place_price <= requested_budget * 1.15:
                score += 5.0
            else:
                score -= 20.0

        if target_close_min is not None:
            hours = str(get_place_attr(place, ["Opening and closing hours", "opening_closing_hours", "Operating Hours", "hours"], ""))
            _, close_min = parse_hours_range(hours)
            if close_min is not None:
                if close_min >= target_close_min:
                    score += 15.0
                else:
                    score -= 20.0

        business_name = str(get_place_attr(place, ["Business name", "business_name", "name"], ""))
        if business_name:
            name_similarity = SequenceMatcher(None, normalize_text(query_text), normalize_text(business_name)).ratio()
            score += name_similarity * 10.0

        scored_places.append({
            "place": place,
            "score": round(score, 2),
        })

    scored_places.sort(key=lambda item: item["score"], reverse=True)
    return scored_places

def get_partial_matches(constraints, session_location):
    if constraints.get("required_features"):
        partial_results = score_and_rank_places_nlp(constraints, session_location, ignore_features=True)
        if partial_results:
            loc_str = constraints.get('location', 'that area').title()
            return partial_results, f"I couldn't find a place matching all features in {loc_str}, but here are the top matching venues nearby:"
    return [], None


# 10. RESPONSE GENERATORS & FORMATTERS
def format_place_output(ranked_results, user_name, has_category=True, custom_prefix=None):
    top_results = ranked_results[:3]

    if not top_results:
        return f"I couldn't find any places matching all your criteria, {user_name}."

    if custom_prefix:
        reply = f"{custom_prefix}<br><br>"
    elif has_category:
        reply = f"Here are the top options matching your query, {user_name}:<br><br>"
    else:
        reply = f"Here are my suggested places for you, {user_name}:<br><br>"

    for idx, item in enumerate(top_results, 1):
        place = item["place"]
        name = get_place_attr(place, ["Business name", "business_name", "name"])
        cat = get_place_attr(place, ["Category", "category"])
        city = get_place_attr(place, ["Municipality or city", "municipality_city", "city"])
        hours = get_place_attr(place, ["Opening and closing hours", "opening_closing_hours", "Operating Hours", "hours"])
        price = get_place_attr(place, ["Price range", "price_range", "Price Range"])

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

    name = get_place_attr(place, ["Business name", "business_name", "name"])
    cat = get_place_attr(place, ["Category", "category"])
    city = get_place_attr(place, ["Municipality or city", "municipality_city", "city"])
    hours = get_place_attr(place, ["Opening and closing hours", "opening_closing_hours", "Operating Hours", "hours"])
    price = get_place_attr(place, ["Price range", "price_range", "Price Range"])
    features_list = get_place_attr(place, ["Features", "features"], default=[])

    features = ", ".join(str(feature) for feature in features_list) if isinstance(features_list, list) else str(features_list)
    description = get_place_attr(place, ["Short description", "short_description", "Description", "description"], default="No description available.")
    maps_url = get_place_attr(place, ["Google Maps", "google_maps", "maps_link", "GoogleMaps", "maps"], default=None)

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

# 11. CONVERSATIONAL FOLLOWUP & SESSION CONTEXT HANDLERS
def find_place_by_name(query_text):
    query_clean = normalize_text(query_text)
    category = detect_category_entity(query_clean)

    search_triggers = ["want", "looking for", "find", "show", "where", "is there", "search", "give me", "recommend"]
    if any(trigger in query_clean for trigger in search_triggers) or category:
        return None

    for place in ALL_PLACES:
        p_name = str(get_place_attr(place, ["Business name", "business_name", "name"], ""))
        p_name_clean = normalize_text(p_name)
        if not p_name_clean:
            continue
            
        if phrase_present(query_clean, p_name_clean):
            return place

    return None

def is_explicit_followup_question(msg_lower):
    # Guard: If the message contains a new location, unsupported town, or category, it is NOT a followup
    if detect_location_entities(msg_lower) or detect_unsupported_location(msg_lower) or detect_category_entity(msg_lower):
        return False

    followup_patterns = [
        r"\bdo they\b", r"\bis there\b", r"\bdoes it\b", r"\bare there\b",
        r"\bhas parking\b", r"\bhave parking\b", r"\bhas wifi\b",
        r"\bhave wifi\b", r"\bhas ktv\b", r"\bhave ktv\b", r"\bwhich one\b",
        r"\bhow about\b", r"\bwhat about\b", r"\band\b", r"\babout\b"
    ]
    return any(re.search(pattern, msg_lower) for pattern in followup_patterns)

def handle_followup_feature_question(msg_lower, last_places, user_name):
    if not is_explicit_followup_question(msg_lower):
        return None

    detected_features = detect_feature_entities(msg_lower)
    if not detected_features:
        return None

    detected_feature = detected_features[0]
    reply = f"Here is the info regarding <b>{detected_feature.title()}</b> for the places listed above, {user_name}:<br><br>"

    for idx, place in enumerate(last_places, 1):
        name = get_place_attr(place, ["Business name", "business_name", "name"])
        if place_has_feature(place, detected_feature):
            reply += f"✅ <b>{idx}. {name}</b>: Yes, feature/info found matching <i>{detected_feature}</i>.<br>"
        else:
            reply += f"❌ <b>{idx}. {name}</b>: No explicit mention of <i>{detected_feature}</i> in details.<br>"

    reply += "<br>Reply with 1, 2, or 3 to view full details for any option!"
    return reply

def get_user_session():
    if "user_data" not in session:
        session["user_data"] = {
            "name": None,
            "location": None,
            "state": STATE_AWAITING_NAME,
            "last_places": [],
            "last_category": None,
            "fsm_context": "IDLE"
        }
    return session["user_data"]

# 12. ROUTING CONTROLLER & DIALOGUE STATE MACHINE (FSM)
@app.route("/")
def home():
    session["user_data"] = {
        "name": None,
        "location": None,
        "state": STATE_AWAITING_NAME,
        "last_places": [],
        "last_category": None,
        "fsm_context": "IDLE"
    }
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json() or {}
    msg = str(data.get("message", "")).strip()
    msg_lower = normalize_text(msg)
    u_session = get_user_session()

    # Step 1: Capture user name if state is AWAITING_NAME
    if u_session["state"] == STATE_AWAITING_NAME or u_session["name"] is None:
        clean_name = re.sub(r"^(hello|hi|hey)?\s*,?\s*(my name is|call me|im|i'm|i am)\s*", "", msg, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r"^(hello|hi|hey)\s+", "", clean_name, flags=re.IGNORECASE).strip()

        u_session["name"] = clean_name.title() if clean_name else "Friend"
        u_session["state"] = STATE_IDLE
        u_session["fsm_context"] = "IDLE"
        session.modified = True

        return jsonify({
            "reply": f"Nice to meet you, {u_session['name']}! What kind of place are you looking for in Ilocos Norte today?"
        })

    display_name = u_session["name"]

    # Step 2: Unsupported Location Guard Clause
    # Immediate polite outofscope feedback if user requests an unindexed municipality
    unsupported_town = detect_unsupported_location(msg_lower)
    if unsupported_town:
        supported_str = ", ".join([m.title() for m in MUNICIPALITIES])
        return jsonify({
            "reply": f"Sorry, {display_name}! I don't cover <b>{unsupported_town}</b> yet. I currently cover places in <b>{supported_str}</b>."
        })

    # Step 3: Direct Single Place Verification Query
    target_place = find_place_by_name(msg)
    detected_features = detect_feature_entities(msg_lower)
    detected_feature = detected_features[0] if detected_features else None

    if target_place and detected_feature and not any(k in msg_lower for k in ["want", "looking for", "find", "show me", "where"]):
        p_name = get_place_attr(target_place, ["Business name", "business_name", "name"])
        if place_has_feature(target_place, detected_feature):
            reply = f"✅ Yes, <b>{p_name}</b> has <b>{detected_feature.title()}</b>, {display_name}!"
        else:
            reply = f"❌ No explicit mention of <b>{detected_feature.title()}</b> for <b>{p_name}</b> in our records, {display_name}."
        return jsonify({"reply": reply})

    # Step 4: Index Selection Routing (User replies "1", "2", or "3")
    if u_session["last_places"] and (msg_lower in ["1", "2", "3", "option 1", "option 2", "option 3", "first", "second", "third"]):
        idx_map = {
            "1": 0, "option 1": 0, "first": 0,
            "2": 1, "option 2": 1, "second": 1,
            "3": 2, "option 3": 2, "third": 2,
        }
        selected_idx = idx_map.get(msg_lower)

        if selected_idx is not None and selected_idx < len(u_session["last_places"]):
            chosen_place = u_session["last_places"][selected_idx]
            u_session["fsm_context"] = "VIEWING_DETAIL"
            session.modified = True
            return jsonify({"reply": format_single_place_detail(chosen_place, display_name)})

    # Step 5: FollowUp Questions on Previous Search Results
    if u_session["last_places"]:
        if is_explicit_followup_question(msg_lower) or (detected_feature and len(normalized_tokens(msg_lower)) <= 4):
            followup_reply = handle_followup_feature_question(msg_lower, u_session["last_places"], display_name)
            if followup_reply:
                u_session["fsm_context"] = "FOLLOWUP"
                session.modified = True
                return jsonify({"reply": followup_reply})

    # Step 6: Conversational Pause & Acknowledgment Triggers
    if msg_lower in ["okay", "ok"]:
        return jsonify({"reply": f"Got it, {display_name}! What place or category would you like to check out next?"})

    if any(phrase in msg_lower for phrase in THINKING_PHRASES):
        u_session["fsm_context"] = "THINKING"
        session.modified = True
        return jsonify({"reply": f"No rush at all, {display_name}! Take your time. Just let me know whenever you're ready to search."})

    if any(phrase in msg_lower for phrase in READY_PHRASES):
        u_session["fsm_context"] = "READY"
        session.modified = True
        return jsonify({"reply": f"Awesome! What kind of place or category would you like to search for, {display_name}?"})

    if msg_lower in GREETINGS:
        return jsonify({"reply": f"Hello {display_name}! How can I help you explore Ilocos Norte right now?"})

    # Step 7: Main NLP Search Pipeline Execution
    constraints = extract_constraints_nlp(msg)
    has_explicit_category = bool(constraints["category"])

    if constraints["category"]:
        u_session["last_category"] = constraints["category"]
        session.modified = True
    elif constraints["locations"] and not constraints["category"] and u_session.get("last_category"):
        constraints["category"] = u_session["last_category"]
        has_explicit_category = True

    if constraints["locations"]:
        u_session["location"] = constraints["locations"][0]
        session.modified = True

    # Multifactor score computation
    ranked_results = score_and_rank_places_nlp(constraints, u_session.get("location"))

    custom_prefix = None
    if not ranked_results:
        ranked_results, custom_prefix = get_partial_matches(constraints, u_session.get("location"))

    # Step 8: Return Top Results and Save State
    if ranked_results:
        u_session["last_places"] = [item["place"] for item in ranked_results[:3]]
        u_session["fsm_context"] = "SEARCH_RESULTS"
        session.modified = True
        return jsonify({"reply": format_place_output(ranked_results, display_name, has_category=has_explicit_category, custom_prefix=custom_prefix)})

    u_session["fsm_context"] = "NO_RESULTS"
    session.modified = True
    return jsonify({"reply": f"I couldn't find any places matching your request, {display_name}."})


# 13. DEBUGGING & ANALYSIS ENDPOINT
def explain_query(query_text):
    """Utility to inspect constraints, tokenization, and NLTK parse trees."""
    constraints = extract_constraints_nlp(query_text)
    return {
        "tokens": constraints["tokens"],
        "category": constraints["category"],
        "locations": constraints["locations"],
        "dish": constraints["dish"],
        "features": constraints["required_features"],
        "max_budget": constraints["max_budget"],
        "target_closing_time": constraints["target_closing_time"],
        "parse_tree": str(constraints["parse_tree"]),
    }

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)