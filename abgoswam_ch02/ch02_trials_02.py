import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

result = re.split(r'([,.]|\s)', text)
print(result)

result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)