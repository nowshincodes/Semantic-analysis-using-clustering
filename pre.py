# Install and import necessary packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset, load_from_disk
import re
import random 
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions

negations = {
    "no", "not", "nor", "never", "neither", "none", "nobody", "nowhere", "nothing", "without",
    "hardly", "barely", "scarcely",
    "isn't", "wasn't", "aren't", "weren't", "won't", "wouldn't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "shouldn't", "mustn't", "mightn't", "haven't", "hasn't", "hadn't",
    "shan't", "needn't",'wouldn','shouldn','weren','isn','against','aren','couldn','didn','doesn','don','hadn','hasn','haven','isn','mightn','mustn',
}
# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words -= negations

# Preprocessing function
def preprocess_text(text):
    
    text = text.lower()                             # Lowercase text
    text = contractions.fix(text)
    text = re.sub(r'\d+', '', text)                 # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)             # Remove punctuation
    text = text.strip()                             # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)                # Remove extra spaces
    tokens = word_tokenize(text)                    # Tokenize the text
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(cleaned)

preprocessed_dir = r"V:/CSE425project/preprocessed_sst2"

if os.path.exists(preprocessed_dir):
    print(f"Loading preprocessed dataset from {preprocessed_dir}...")
    train_dataset = load_from_disk(preprocessed_dir)
else:
    print("Preprocessing dataset...")
    dataset = load_dataset("glue", "sst2")
    t_dataset = dataset['train']
    # positive_dataset = dataset["train"].filter(lambda example: example["label"] == 1)

    # Randomly select 500 indices from the positive dataset
    indices = random.sample(range(len(t_dataset)), 500)

    # Select those 500 positive examples
    train_dataset = t_dataset.select(indices)
    print(print(train_dataset["sentence"][:10]))
    # Define preprocessing function for batched processing
    def preprocess_batch(batch):
        return {"sentence": [preprocess_text(s) for s in batch["sentence"]]}

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_batch, batched=True, batch_size=1000)
    
    # Save preprocessed dataset to disk
    train_dataset.save_to_disk(preprocessed_dir)
    print(f"Preprocessed dataset saved to {preprocessed_dir}")

print(train_dataset["sentence"][:10])