# encoding: utf-8

import json

x = []

with open("wikipediaData.json", "r+") as f:
    x = json.load(f)

temp = ''
while(1):
    try:
        t = input()
        temp += ' ' + t
    except EOFError:
        x.append(temp)
        break

with open("wikipediaData.json", "w+") as f:
    json.dump(x, f, indent=4)
