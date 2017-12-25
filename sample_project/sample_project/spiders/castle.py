import re

a = "Header"
b = "heaDer"
c = "headeR"
pattern_header = re.search("^header", a , re.IGNORECASE)
print(pattern_header)
pattern_header = re.search("^header", b , re.IGNORECASE)
print(pattern_header)
pattern_header = re.search("^header", c , re.IGNORECASE)
print(pattern_header)
