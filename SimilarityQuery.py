# Front-End program to do similarity queries

from SimilarityFunction import search_similarity_query

text = input("Type the text that will be used:\n\t")
print()

n = int(input("Type how many most similar documents will be shown:\n\t"))
print()

result = search_similarity_query(text, n)

if result.empty:
    print("There were no similar subjects to your inserted text\n")

else:
    print(result)