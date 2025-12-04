"""
PII Lexicon - all the terms we detect and redact.
Days, months, colors, states, and cities.
"may" is handled separately with context rules.
Cities with color names (Brownsville, Greenville) need to match as cities first.
"""

from typing import Dict, List, Set

# Days of the week (+ common abbreviations)
DAYS: List[str] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"
]

# Months (excluding "may" - handled with context rules)
MONTHS: List[str] = [
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
]

# Colors (common color names)
COLORS: List[str] = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "gold", "silver",
    "violet", "indigo", "teal", "cyan", "magenta", "maroon", "navy",
    "beige", "tan", "coral", "turquoise", "lavender", "crimson",
    "amber", "aqua", "bronze", "burgundy", "charcoal", "chartreuse",
    "chocolate", "copper", "cream", "fuchsia", "ivory", "jade",
    "khaki", "lilac", "lime", "mauve", "olive", "peach", "periwinkle",
    "plum", "rose", "ruby", "salmon", "sapphire", "scarlet", "sienna",
    "slate", "taupe", "topaz", "vermillion"
]

# US States (full names only - no abbreviations found in data)
STATES: List[str] = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
    # DC
    "district of columbia"
]

# Cities - Multi-word (MUST match before single-word to avoid partial matches)
# Match cities before colors to avoid "Brownsville" -> "[COLOR]sville"
CITIES_MULTI: List[str] = [
    # 3-word cities
    "salt lake city", "new york city", "oklahoma city", "kansas city",
    "virginia beach",
    # 2-word cities
    "new york", "los angeles", "san francisco", "san diego", "san antonio",
    "san jose", "las vegas", "fort worth", "el paso", "new orleans",
    "long beach", "colorado springs", "st louis", "st paul", "st petersburg",
    "santa fe", "santa ana", "santa monica", "baton rouge", "little rock",
    "grand rapids", "des moines", "ann arbor", "corpus christi",
    "round rock", "college station"
]

# Cities - Single word
CITIES_SINGLE: List[str] = [
    # Major US cities
    "houston", "phoenix", "philadelphia", "dallas", "austin", "jacksonville",
    "charlotte", "seattle", "denver", "boston", "detroit", "portland",
    "memphis", "baltimore", "milwaukee", "albuquerque", "tucson", "fresno",
    "sacramento", "atlanta", "miami", "oakland", "minneapolis", "cleveland",
    "tulsa", "pittsburgh", "cincinnati", "indianapolis", "nashville",
    "chicago", "omaha", "raleigh", "richmond", "buffalo", "orlando",
    "tampa", "honolulu", "anchorage",
    # Texas cities (from data)
    "arlington", "irving", "garland", "mesquite", "plano", "waco",
    "lubbock", "amarillo", "laredo", "midland", "odessa", "brownsville",
    "mcallen", "killeen", "pasadena", "beaumont", "abilene", "carrollton",
    "frisco", "lewisville", "denton", "richardson", "tyler",
    "pearland", "conroe", "edinburg",
    # Other from data
    "burkburnett", "hilton", "lincoln", "savannah", "mobile",
    # Cities containing color names (match as cities, not colors)
    "greenville", "greensboro", "brownwood", "blacksburg", "whitehall",
    "goldsboro", "silverdale", "bluefield", "redmond", "redding",
    "orangeburg", "pinkville"
]

# Category labels for redaction
CATEGORY_LABELS: Dict[str, str] = {
    "day": "[DAY]",
    "month": "[MONTH]",
    "color": "[COLOR]",
    "state": "[STATE]",
    "city": "[CITY]"
}

def get_all_pii_terms() -> Dict[str, Set[str]]:
    """Return all PII terms organized by category."""
    return {
        "day": set(DAYS),
        "month": set(MONTHS),
        "color": set(COLORS),
        "state": set(STATES),
        "city": set(CITIES_MULTI + CITIES_SINGLE)
    }

def get_sorted_terms_by_length() -> List[tuple]:
    """
    Return all PII terms sorted by length (longest first).
    This ensures "New York City" matches before "New York" before "New".

    Cities are checked before colors to prevent
    "Brownsville" from becoming "[COLOR]sville".

    Returns:
        List of (term, category) tuples sorted by term length descending.
    """
    all_terms = []

    # Add in priority order: cities first (to catch color-containing city names)
    for term in CITIES_MULTI:
        all_terms.append((term, "city"))
    for term in CITIES_SINGLE:
        all_terms.append((term, "city"))

    # Then states (multi-word states like "New York" the state)
    for term in STATES:
        all_terms.append((term, "state"))

    # Then other categories
    for term in DAYS:
        all_terms.append((term, "day"))
    for term in MONTHS:
        all_terms.append((term, "month"))
    for term in COLORS:
        all_terms.append((term, "color"))

    # Sort by length (longest first), then alphabetically for stability
    return sorted(all_terms, key=lambda x: (-len(x[0]), x[0]))
