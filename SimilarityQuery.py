# Front-End program to do similarity queries

from SimilarityFunction import search_similarity_query

text = input("Type the text that will be used:\n\t")
print()

n = int(input("Type how many most similar documents will be shown:\n\t"))
print()

# TODO: It's probably better to insert the threshold as function parameters separate into 2 function calls: 1 to n-best and other for >= threshold
t = float(input("Would you like to set up a minimum value? [0, 1], type -1 for no:\n\t"))
print()

result = search_similarity_query(text, n)

if (t != -1):
    result = result[result['Relevancia'] >= t]

if result.empty:
    print("There were no similar subjects to your inserted text or selected threshold\n")
else:
    print(result)