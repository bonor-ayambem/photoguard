import warnings
warnings.simplefilter("ignore")

from PIL import Image
import matplotlib.pyplot as plt
import os
import csv
import json
import time

import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import random

import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util

def load_images():
    image_dir = 'flickr8k/Images'
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)]

    image_paths.sort()

    output_dir = 'original-images'
    os.makedirs(output_dir, exist_ok=True)

    for id, path in enumerate(image_paths):
        # Open the image
        image = Image.open(path)
        output_path = os.path.join(output_dir, f'{id + 1}.jpg')
        image.save(output_path)


    csv_file_path = 'flickr8k/captions.txt'
    json_file_path = 'flickr8k/captions.json'

    captions = {}

    # Read the CSV file and create a dictionary
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        # Sort rows based on the 'image' column
        sorted_rows = sorted(csvreader, key=lambda x: x['image'])

        for row in sorted_rows:
            image_filename = row['image']
            caption = row['caption']
            captions[image_filename] = caption

    captions = {str(i + 1): value for i, (key, value) in enumerate(captions.items())}


    # Write the dictionary to a JSON file
    with open(json_file_path, 'w') as jsonfile:
        json.dump(captions, jsonfile, indent=2)

    print(f'JSON file "{json_file_path}" created successfully.')

def find_closest_bert(pos, options, orig_word, sentence):
    # print(orig_word)
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

    bert_preprocess_model = hub.KerasLayer(preprocess_url)
    text_preprocessed = bert_preprocess_model([sentence])

    bert_model = hub.KerasLayer(encoder_url)
    bert_results = bert_model(text_preprocessed)

    original_word_encoding = bert_results['pooled_output'][0].numpy()

    # print(original_word_encoding)

    differences = {}

    for op in options:
        word = op.name()
        word = word.split(".")[0].replace('_', ' ')
        if word == orig_word:
            continue
        sentence_words = sentence.split()
        sentence_words[pos] = word
        new_sentence = " ".join(sentence_words)
        # print(new_sentence)
        preprocessed = bert_preprocess_model([new_sentence])
        results = bert_model(preprocessed)

        # print(results['pooled_output'][0].numpy())

        diff = cosine_similarity(results['pooled_output'][0].numpy().reshape(1, -1) - original_word_encoding.reshape(1, -1))
        # print(diff)

        differences[op] = diff[0, 0]

    # minimum_diff = min(differences.values())

    sorted_differences = dict(sorted(differences.items(), key=lambda item: item[1]))
    limit = int(0.1 * len(sorted_differences))
    print(sorted_differences)
    print("limit is {}", limit)
    return random.choice(list(sorted_differences.items())[-(limit+1):])[0]
    # for key, value in differences.items():
    #     if value == minimum_diff:
    #         print(key)
    #         exit()

def compare_embeddings(pos, options, orig_word, sentence):
    print(sentence)
    print(orig_word)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reference = model.encode([sentence], convert_to_tensor=True)

    new_sentences = {} # new word --> embedding

    for op in options:
        word = op.name()
        word = word.split(".")[0].replace('_', ' ')
        if word == orig_word:
            continue
        sentence_words = sentence.split()
        # print(op, sentence_words, pos)
        sentence_words[pos] = word
        new_sentence = " ".join(sentence_words)

        new_sentences[op] = model.encode([new_sentence], convert_to_tensor=True)

    similarities = {}
    for word, emb in new_sentences.items():
        similarities[word] = util.cos_sim(reference, emb)

    sorted_sims = dict(sorted(similarities.items(), key=lambda item: item[1]))
    limit = int(0.1 * len(sorted_sims))

    print(sorted_sims)
    # print("limit is ", limit)
    if(len(sorted_sims) == 0):
        return list(options)[0]
    new_word = random.choice(list(sorted_sims.items())[-(limit+1):])[0]
    print(new_word)

    return new_word
    # if(len(sorted_sims) == 0):
    #     return list(options)[0]
    # return list(sorted_sims.keys())[-1]


def replace_nouns():
    from nltk.tag import pos_tag

    new_captions = {}
    data = {}

    # Open the JSON file
    with open('flickr8k/captions.json', 'r') as file:
        # Load the JSON data
        data = json.load(file)

    for id, (key, caption) in enumerate(data.items()):
        print(id)
        words = word_tokenize(caption)
        pos_tags = pos_tag(words)

        new_caption_words = []
        nouns = set()
        for word, tag in pos_tags:
            if tag.startswith('N'):
                nouns.add(word)
            
        # if len(nouns) < :
        #     break
        
        nouns_to_change = set()
        if len(nouns) > 0:
            nouns_to_change.add(random.choice(list(nouns)))
        nouns.remove(list(nouns_to_change)[0])
        if len(nouns) > 0:
            nouns_to_change.add(random.choice(list(nouns)))
        for id2, (word, tag) in enumerate(pos_tags): #could replace pos_tag w words
            if word in nouns_to_change: 
                # print("word is ", word)
                synsets = wn.synsets(word, pos=wn.NOUN)
                # if len(synsets) > 0:
                #     hypernyms = synsets[0].hypernyms()
                # else:
                #     continue
                
                # if len(hypernyms) > 0:
                #     hyponyms = hypernyms[0].hyponyms()
                # else:
                #     continue

                hypernyms = set()
                # print("synsets are ", synsets)

                # print(synsets)
                if len(synsets) > 0:
                    for syn in synsets:
                        # print(syn.name().split('.')[0].replace('_', ' '))
                        if(syn.name() == synsets[0].name()):
                        # hypernyms.add(syn)
                            # print("taking", syn)
                            for hyper in syn.hypernyms():
                                # print(hyper)
                                hypernyms.add(hyper)
                else:
                    continue

                all_hyponyms = set()

                for hyper in hypernyms:
                    # print(hyper)
                    # print(hyper.hyponyms())
                    for hypo in hyper.hyponyms():
                        all_hyponyms.add(hypo)
                
                # print(hypernyms)
                # print(all_hyponyms)
                        
                if(len(all_hyponyms) == 0):
                    all_hyponyms.add(synsets[0])
                # random_synset = hyponyms[0]
                random_synset = compare_embeddings(id2, all_hyponyms, word, caption)
                new_word = random_synset.name()
                new_word = new_word.split('.')[0].replace('_', ' ')
                if word.lower() == 'girl':
                    new_word = 'boy'
                if word.lower() == 'woman':
                    new_word = 'man'
                if word.lower() == 'boy':
                    new_word = 'girl'
                if word.lower() == 'man':
                    new_word = 'woman'
                if word.lower() == 'dog':
                    new_word = 'cat'
                if word.lower() == 'young':
                    new_word = 'old'
            else:
                new_word = word  # If unable to determine POS, keep the original word
            new_caption_words.append(new_word)
            
        new_captions[key] = ' '.join(new_caption_words)
        print(new_captions[key])
        print('\n\n')
        if id == 800:
            break

    # print(new_captions)
    # json_file_path = 'flickr8k/second_captions.json'
    # # Write the dictionary to a JSON file
    # with open(json_file_path, 'w') as jsonfile:
    #     json.dump(new_captions, jsonfile, indent=2)
    # print(f'JSON file "{json_file_path}" created successfully.')
        
def replace_nouns_gemini():
    import textwrap
    import google.generativeai as genai
    from IPython.display import display
    from IPython.display import Markdown

    genai.configure(api_key='AIzaSyA0lUyXAjlSKuvy-rpLgXxpnbFt7XqLip0')

    model = genai.GenerativeModel('gemini-pro')

    new_captions = {}
    data = {}
    not_generated = []

    json_file_path = 'flickr8k/gemini_captions_7.json'

    # Open the JSON file
    with open('flickr8k/captions.json', 'r') as file:
        # Load the JSON data
        data = json.load(file)

    for id, (key, caption) in enumerate(data.items()):
        if int(key) <= 6333:
            continue

        prompt_string = "Rewrite the following sentence but select and change only one of its nouns and/or adjectives with a different one that is commonly used and fits contextually. For example, change colors, genders, ages, animals, etc with different ones: " + caption
        # response = model.generate_content(prompt_string)

        # print(response.candidates)
        # while not response.parts:
        #     response = model.generate_content(prompt_string)
                # Attempt to generate content until successful
        attempts = 10
        while True:
            attempts = attempts - 1
            try:
                response = model.generate_content(prompt_string)
                print(response.candidates)
                if response.candidates:  # Ensures there are candidates before proceeding
                    new_captions[key] = response.text
                    break
                elif attempts == 0:
                    not_generated.append(key)
                    new_captions[key] = caption
                    break            
            except ValueError as e:
                print(f"Error: {e}. Retrying...")
            time.sleep(1)

        # Write the dictionary to a JSON file
        with open(json_file_path, 'w') as jsonfile:
            json.dump(new_captions, jsonfile, indent=2)

        # if id == 800:
        #     break
        # else:
        print(key)
        time.sleep(5) #5875
    
        
    # print(f'JSON file "{json_file_path}" created successfully.')

    print("The following captions were not changed:", not_generated)

    
if __name__ == "__main__":
    # replace_nouns_gemini()

    not_changed = []

    originals = {}
    gemini = {}

    with open('flickr8k/captions.json', 'r') as file:
       originals = json.load(file)

    with open('flickr8k/gemini_captions.json', 'r') as file:
       gemini = json.load(file)

    for key, caption in originals.items():
        if originals[key] == gemini[key]:
            not_changed.append(key)
    
    print("The following indices were not changed: ", not_changed)

    # The following indices were not changed:  ['336', '1176', '1373', '1561', '1841', '2180', '2731', '2815', '3611', '3798', '4162', '5050', '5804', '5901', '6817', '6918', '7297', '7655', '7870', '7988']
