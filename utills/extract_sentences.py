import pickle
import csv

# Load the dictionary from the file
with open('dataset/Huth/2.pca1000.wq.pkl.dic', 'rb') as f:
    data = pickle.load(f)

# Process the data to reformat the 'words' content
output_data = {}
for key, value in data.items():
    if 'words' in value:
        # Flatten each list of words into a single sentence
        sentences = [' '.join(words) for words in value['words']]
        # Combine all sentences into one text
        formatted_text = ' '.join(sentences)
        output_data[key] = formatted_text

# Save the reformatted data to a CSV file
with open('dataset_text/test_sentence.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Key', 'FormattedText'])
    for key, formatted_text in output_data.items():
        writer.writerow([key, formatted_text])

print("Data reformatted and saved to 'reformatted_data.csv'")
