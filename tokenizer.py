import re
from corpus import raw_text



preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

if __name__ == "__main__":
    print(len(preprocessed))
    print(preprocessed[:30])