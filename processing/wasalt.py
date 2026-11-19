"""Clean one raw Wasalt listing into the common record shape.

The raw listing is the JSON Wasalt embeds in its property page. The parts we use:
  propertyInfo          title, price, sale/rent, type, city, district, description
  attributes            bedrooms, bathrooms, area
  additionalAttributes  extra facts (facade, age, street width...) plus many
                        administrative ones (licence dates, plot borders) we skip
  location              latitude / longitude
  classificationData    names of the listing's photos
"""

from processing.record import new_record
from processing.text import html_to_text, tidy, to_int, to_number

# Extra facts worth keeping (key in Wasalt's data -> label we show).
USEFUL_FACTS = {
    "propertyFacade": "Facade",
    "completionYear": "Age",
    "propertyStreetWidth": "Street width",
    "noOfFloors": "Total floors",
    "floorNumber": "Floor",
    "noOfParkings": "Parking spaces",
    "noOfLivingrooms": "Living rooms",
    "noOfGuestrooms": "Guest rooms",
    "noOfApartments": "Apartments in building",
}

# Wasalt's English city names are sometimes odd transliterations.
CITY_FIXES = {
    "Makkah Al Mukarramah": "Makkah",
    "Aldammam": "Dammam",
    "Alttayif": "Taif",
    "Earear": "Arar",
    "Bariduh": "Buraydah",
    "Alkhubar": "Khobar",
    "Al Madinah Al Munawwarah": "Madinah",
}

IMAGE_URL = "https://imagedelivery.net/1DNKFJPRaeUdy_j8F7HT3w/production/properties/{id}/images/{name}/quality=50,w=480"


def pick_image(listing: dict) -> str | None:
    """Build a URL for one listing photo, preferring a photo of the outside."""
    photos = listing.get("classificationData") or []
    if not photos:
        return None
    facade = [p for p in photos if p.get("classificationName") == "facade"]
    photo = (facade or photos)[0]
    return IMAGE_URL.format(id=listing["id"], name=photo["name"])


def rent_period(info: dict) -> tuple[str | None, float | None]:
    """Return (period, amount) of a rental, e.g. ("yearly", 54500).

    Wasalt stores it as {"yearly": {"amount": 54500, "default_freq": True}, ...};
    a listing can offer several periods, so we take the default one.
    """
    periods = info.get("rentFreq") or {}
    if not isinstance(periods, dict) or not periods:
        return None, None
    period, details = next(
        ((name, d) for name, d in periods.items() if d.get("default_freq")),
        next(iter(periods.items())),
    )
    return period, to_number(details.get("amount"))


def clean_listing(raw: dict) -> dict:
    """Convert one line of data/raw/wasalt.jsonl into a clean record."""
    listing = raw["listing"]
    info = listing["propertyInfo"]
    numbers = {a["key"]: a.get("value") for a in listing.get("attributes") or []}
    extras = {a["key"]: a.get("value") for a in listing.get("additionalAttributes") or []}

    purpose = info.get("propertyFor")  # "sale" or "rent"
    period, rent_amount = rent_period(info) if purpose == "rent" else (None, None)
    if purpose == "sale":
        price = to_number(info.get("salePrice"))
    else:
        price = rent_amount or to_number(info.get("expectedRent"))

    facts = {}
    for key, label in USEFUL_FACTS.items():
        value = extras.get(key) or numbers.get(key)
        if value not in (None, "", "-"):
            facts[label] = tidy(str(value))

    city = info.get("city")
    location = listing.get("location") or {}

    return new_record(
        id=f"wasalt-{listing['id']}",
        source="Wasalt",
        kind="listing",
        title=tidy(info.get("title") or ""),
        purpose=purpose,
        property_type=info.get("propertySubType"),
        usage=info.get("propertyMainType"),
        country=info.get("country"),
        city=CITY_FIXES.get(city, city),
        district=info.get("district"),
        location_text=info.get("address"),
        price=price or to_number(info.get("conversionPrice")),
        currency=info.get("currencyType") or "SAR",
        price_period=period,
        bedrooms=to_int(numbers.get("noOfBedrooms")),
        bathrooms=to_int(numbers.get("noOfBathrooms")),
        area_sqm=to_number(numbers.get("builtUpArea")) or to_number(numbers.get("carpetArea")),
        furnishing=extras.get("furnishingType") or info.get("furnishingType"),
        facts=facts,
        description=html_to_text(info.get("description")),
        url=raw["url"],
        image_url=pick_image(listing),
        latitude=to_number(location.get("lat")),
        longitude=to_number(location.get("lon")),
        scraped_at=raw["scraped_at"],
    )
