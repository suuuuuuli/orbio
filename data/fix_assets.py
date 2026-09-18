"""Naprawa assets.json: plik ma dwa bloki tokenow (stary i nowy, wklejony) i
brakujacy przecinek miedzy nimi. Zostawiamy NOWE wpisy, ale przenosimy z
starych seed_urls, ktorych nowe nie maja - to sprawdzone zrodla do pakietu.
"""
import json
import pathlib
import re

p = pathlib.Path(__file__).resolve().parent / "assets.json"
raw = p.read_text(encoding="utf-8")

# Brakujacy przecinek: "}" nowa linia '  "TICKER": {'
fixed = re.sub(r"\}\s*\n(\s*)\"([A-Z]+)\": \{", r"},\n\1\"\2\": {", raw)

# Duplikaty kluczy: pierwszy wpis to stary, ostatni to nowy (wklejony na koncu).
old: dict[str, dict] = {}
new: dict[str, dict] = {}


def hook(pairs):
    out = {}
    for key, value in pairs:
        if key in out:                      # duplikat na poziomie tickerow
            old[key] = out[key]
            new[key] = value
        out[key] = value
    return out


data = json.loads(fixed, object_pairs_hook=hook)

przeniesione = []
for ticker, previous in old.items():
    entry = data[ticker]
    for key in ("seed_urls", "defillama_slug", "defillama_chain"):
        if key in previous and key not in entry:
            entry[key] = previous[key]
            przeniesione.append(f"{ticker}.{key}")

p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"tickery: {len(data)} ({sum(1 for e in data.values() if e.get('kind') == 'token')} tokenow)")
print(f"nadpisane duplikaty: {sorted(old) or 'brak'}")
print(f"przeniesione ze starych wpisow: {przeniesione or 'brak'}")
print(f"bez seed_urls: {[t for t, e in data.items() if not e.get('seed_urls')]}")
