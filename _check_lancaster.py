import nltk.stem.lancaster as l
import re

src = open(l.__file__).read()
matches = re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*\)', src)
print(f"Total NLTK Lancaster rules: {len(matches)}")
print(f"Our rules: 113")

for i, m in enumerate(matches):
    if i >= 113:
        suffix, repl, cont = m
        print(f"  [{i}] (\"{suffix}\", \"{repl}\", {cont})")

# Group by continuation class
classes = {}
for m in matches:
    c = int(m[2])
    classes.setdefault(c, []).append(m)
for k in sorted(classes.keys()):
    print(f"  Class {k}: {len(classes[k])} rules")
