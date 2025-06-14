from convokit import Corpus, download

corpus = Corpus(filename=download("movie-corpus"))
from textblob import TextBlob
import text2emotion as te
import nltk
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


# Download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# Load utterances DataFrame (assuming `corpus` is already loaded)
utterances_df = corpus.get_utterances_dataframe().copy()

# Define the persona types
PERSONAS = ["Comedic Relief Friend", "Wise Mentor", "Skeptical Realist", "Optimistic Dreamer"]

# Sentiment-based logic
def get_sentiment_persona(dialogue):
    blob = TextBlob(dialogue)
    polarity = blob.sentiment.polarity
    if polarity > 0.4:
        return "Optimistic Dreamer"
    elif polarity < -0.2:
        return "Skeptical Realist"
    elif 0.2 < polarity <= 0.4:
        return "Wise Mentor"
    else:
        return "Comedic Relief Friend"

# Emotion-based logic
def get_emotion_persona(dialogue):
    emotions = te.get_emotion(dialogue)
    if emotions['Happy'] > 0.4:
        return "Optimistic Dreamer"
    elif emotions['Angry'] > 0.3 or emotions['Fear'] > 0.3:
        return "Skeptical Realist"
    elif emotions['Surprise'] > 0.3:
        return "Comedic Relief Friend"
    elif emotions['Sad'] > 0.3:
        return "Wise Mentor"
    return max(emotions, key=emotions.get)

# Rule-based classification
def rule_based_persona(dialogue):
    dialogue_lower = dialogue.lower()
    if any(word in dialogue_lower for word in ['believe', 'dream', 'future', 'someday', 'hope']):
        return "Optimistic Dreamer"
    elif any(word in dialogue_lower for word in ['realistic', 'not work', 'won’t', 'impossible', "odds", "against us"]):
        return "Skeptical Realist"
    elif any(word in dialogue_lower for word in ['life', 'truth', 'lesson', 'understand', 'wisdom']):
        return "Wise Mentor"
    elif any(word in dialogue_lower for word in ['lol', 'kidding', 'great', 'wow', 'sarcasm', 'cactus']):
        return "Comedic Relief Friend"
    return "Comedic Relief Friend"

# Combined classification logic (no print)
def classify_dialogue(dialogue):
    persona_sent = get_sentiment_persona(dialogue)
    persona_emot = get_emotion_persona(dialogue)
    persona_rule = rule_based_persona(dialogue)

    votes = [persona_sent, persona_emot, persona_rule]
    final = max(set(votes), key=votes.count)
    return final

# Add the 'predicted_persona' column and initialize it
utterances_df['predicted_persona'] = None

# Apply classification to the entire dataset
total = len(utterances_df)
save_step = int(total * 0.05)
output_path = "/content/drive/MyDrive/Data_set/persona_classified_utteranutterances_dces.csv"

# Classification with checkpointing
for i in tqdm(range(save_step+1,total)):
    # Use .iloc for integer-location based indexing
    text = str(utterances_df.iloc[i, utterances_df.columns.get_loc('text')])
    utterances_df.iloc[i, utterances_df.columns.get_loc('predicted_persona')] = classify_dialogue(text)

    if (i + 1) % save_step == 0 or i == total - 1:
        utterances_df.to_csv(output_path, index=False)
        print(f"✅ Saved at {((i + 1) / total) * 100:.1f}%: {output_path}")