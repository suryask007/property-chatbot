"""Module 2 - Cleaning.

Turns the raw scraped files in data/raw/ into one list of properties with the
same fields for both sources, saved to data/clean/properties.json.

  * processing/wasalt.py    - one Wasalt listing  -> one clean record
  * processing/darglobal.py - one DarGlobal project -> one clean record
  * processing/build.py     - runs both, removes duplicates, writes the output
"""
