"""Fix the block-cursor glyph: CSS needs a hex escape (\\258C ), not \\u258C."""
import pathlib
import re

p = pathlib.Path(__file__).resolve().parent / "index.html"
s = p.read_text(encoding="utf-8")

BAD = 'content: "\\u258C";'
GOOD = 'content: "\\258C ";'          # backslash + hex + terminating space
COMMENT = "          /* block cursor; CSS hex escape, not \\u258C */"

line = re.search(r"[ \t]*content: \"\\\\u258C\";[^\n]*", s)
if line is None:
    raise SystemExit("cursor rule not found")

s = s.replace(line.group(0), "  " + GOOD + COMMENT, 1)
p.write_text(s, encoding="utf-8")

rule = re.search(r"\.bubble\.typing \.said::after \{(.*?)\}", s, re.S).group(0)
value = re.search(r'content: "(.*?)"', rule).group(1)
print(rule.strip())
print("content bytes:", [hex(ord(c)) for c in value])
print("valid CSS escape:", value.startswith("\\") and value[1:].strip().upper() == "258C")
