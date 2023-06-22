# Count the number of words in a sentence
def count_words(sentence):
    words = sentence.split()
    return len(words)

sentence = "This is a sample sentence"
word_count = count_words(sentence)
print("Number of words:", word_count)
