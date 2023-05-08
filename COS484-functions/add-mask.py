import re

def find_and_replace_numeral(sentence, numeral_words):
    for word in sentence.split():
        if word.lower() in numeral_words:
            sentence = sentence.replace(word, "<mask>", 1)
            sentence += '\t' + word
            break
    return sentence

def process_sentences(input_file, output_file):
    numeral_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero", "no"]

    with open(input_file, "r") as input_file:
        sentences = input_file.readlines()

    with open(output_file, "w") as output_file:
        for sentence in sentences:
            modified_sentence = find_and_replace_numeral(sentence.strip(), numeral_words)
            output_file.write(modified_sentence + "\n")

process_sentences("COS484-data/gkb_best_filtered.txt", "COS484-data/gkb_best_filtered.tsv")