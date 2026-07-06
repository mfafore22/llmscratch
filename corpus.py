import os
import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

filepath = "the-verdict-txt"

if not os.path.exists(filepath):
    urllib.request.urlretrieve(url, filepath)

with open(filepath, "r", encoding="utf-8") as f:
    raw_text = f.read()