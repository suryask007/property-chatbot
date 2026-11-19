"""The one shape every clean property record has, whatever website it came from."""


def new_record(**values) -> dict:
    """Return a property record with every field present (unknown ones are None).

    Fields
    ------
    id             unique id, e.g. "wasalt-5841667" or "darglobal-the-astera"
    source         "Wasalt" or "DarGlobal"
    kind           "listing" (one property for sale/rent) or "project" (a development)
    title          name shown on the website
    purpose        "sale" or "rent" (None for DarGlobal projects)
    property_type  e.g. "Apartment", "Villa", "Luxury Villas"
    usage          e.g. "Residential", "Commercial"
    country, city, district
    location_text  full location as written on the website
    price          number, or None when the website shows no price
    currency       e.g. "SAR"
    price_period   for rentals: e.g. "yearly", "monthly"
    bedrooms       number of bedrooms (listings)
    bathrooms      number of bathrooms (listings)
    area_sqm       built-up area in square metres (listings)
    unit_types     text such as "1 to 3-bedroom residences" (projects)
    area_text      text such as "680 - 1,515 sqm" (projects)
    status         e.g. "Under Development", "Sold Out" (projects)
    furnishing     e.g. "Furnished" (listings)
    amenities      list of strings
    facts          dict of other useful facts, e.g. {"Facade": "North"}
    description    plain-text description
    faqs           list of {"question", "answer"} (projects)
    url            link to the original page
    image_url      one picture for the result card
    latitude, longitude
    scraped_at     when the page was scraped
    """
    record = {
        "id": None, "source": None, "kind": None, "title": None,
        "purpose": None, "property_type": None, "usage": None,
        "country": None, "city": None, "district": None, "location_text": None,
        "price": None, "currency": None, "price_period": None,
        "bedrooms": None, "bathrooms": None, "area_sqm": None,
        "unit_types": None, "area_text": None, "status": None, "furnishing": None,
        "amenities": [], "facts": {}, "description": "", "faqs": [],
        "url": None, "image_url": None, "latitude": None, "longitude": None,
        "scraped_at": None,
    }
    unknown = set(values) - set(record)
    if unknown:
        raise KeyError(f"Not a record field: {unknown}")
    record.update(values)
    return record
