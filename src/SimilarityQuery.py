# Front-End program to do similarity queries

import SimilarityFunction
import os

while(1):
    filename = input("Type the name of the file that will provide the subjects (without extension)\n\t")
    if os.path.isfile(filename + '.json'):
        break
    else:
        print('Invalid file, check if the file exists and is typed correctly!\n')
print()

text = input("Type the text that will be used in the query :\n\t")
print()

n = int(input("Type how many most similar documents will be shown:\n\t"))
print()

# TODO: Maybe it's better to use the threshold in a second function. Therefore, 2 functions will be created: one to n-best and other for >= threshold
t = float(input("Would you like to set up a minimum value? [0, 1], type -1 if you don't:\n\t"))
print()

result = SimilarityFunction.search_similarity_query(filename, text, n)

if (t != -1):
    result = result[result['Relevancia'] >= t]

if result.empty:
    print("There were no similar subjects to your inserted text or selected threshold\n")
else:
    print(result)