import torch
import numpy as np
from transformers import AutoTokenizer
from model import MultiTaskModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hing-bert")
try:
    target_classes = np.load("saved_model/target_classes.npy", allow_pickle=True)
    NUM_TARGETS = len(target_classes)
except FileNotFoundError:
    print("Error: Could not find target_classes.npy. Please run train.py first.")
    exit()
model = MultiTaskModel(num_targets=NUM_TARGETS).to(device)
try:
    model.load_state_dict(torch.load("saved_model/model.pt", map_location=device))
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
model.eval()
while True:
    text = input("\nEnter Hinglish comment (or type 'quit' to exit): ")
    if text.lower() == 'quit':
        break
    if not text.strip():
        continue
    encoding = tokenizer(
        text, max_length=128, padding="max_length", 
        truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        bully_logits, target_logits, rationale_logits = model(input_ids, attention_mask)
    bully_pred = torch.argmax(bully_logits, dim=1).item()
    target_pred = torch.argmax(target_logits, dim=1).item()
    bully_label_text = "Bully" if bully_pred == 1 else "Non-Bully"
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    rationale_probs = torch.softmax(rationale_logits, dim=2)[0]
    toxic_probabilities = rationale_probs[:, 1]
    rationale_preds = (toxic_probabilities > 0.50).long()
    raw_toxic_words = []
    current_word = ""
    for token, pred in zip(tokens, rationale_preds):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue   
        if pred.item() == 1:
            if token.startswith("##"):
                current_word += token.replace("##", "")
            else:
                if current_word:
                    raw_toxic_words.append(current_word)
                current_word = token
        else:
            if current_word:
                raw_toxic_words.append(current_word)
                current_word = ""               
    if current_word:
        raw_toxic_words.append(current_word)
    stopwords = {"tu", "tum", "hai", "ho", "kya", "ki", "ka", "ke", "ye", "wo", "se", "ko", "is", "the", "a"}
    toxic_words = [w for w in raw_toxic_words if w.lower() not in stopwords and len(w) > 1]
    print("\n" + "="*45)
    print(f"🧐 PREDICTION RESULTS")
    print("="*45)
    print(f"Text:          {text}")
    print(f"Bully Status:  {bully_label_text}")
    print(f"Target Class:  {target_classes[target_pred]}")    
    if bully_pred == 1 and toxic_words:
        highlighted = " ".join(toxic_words)
        print(f"Rationale:     Identified toxic words -> [{highlighted}]")
    elif bully_pred == 1 and not toxic_words:
        print(f"Rationale:     Model classified as Bully, but couldn't pinpoint specific toxic words.")
    else:
        print(f"Rationale:     No toxic words found.")
    print("="*45)