import re
text = "San Jose, CA"
#text = "Carolina"
result = re.findall("\,\s[A-Z]{2}", text)
print(result[0][2:])