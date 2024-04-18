import pickle
import json
import re





def clean_string(input_string):
    # Remove new lines, multiple spaces, and tabs
    cleaned_string = re.sub(r'\s+', ' ', input_string.replace('\n', ' ').strip())
    return cleaned_string


metadata = None


def get_metadata():
    global metadata
    if metadata is None:
        with open("dataset/lsipcr-metadata-sm.json", "r") as f:
            metadata = json.load(f)
    return metadata




#------------------------------------------------------------Dataset------------------------------------------------------------

precedients = None

def get_precedients():
    global precedients
    if precedients is None:
        with open("dataset/lsipcr-precs.json", "r") as f:
            precedients = json.load(f)
    return precedients


precedients_anon = None

def get_precedients_anon():
    global precedients_anon
    if precedients_anon is None:
        with open("dataset/lsipcr-precs-anon.json", "r") as f:
            precedients_anon = json.load(f)
    return precedients_anon



precedients_triplets = None

def get_precedients_triplets():
    global precedients_triplets
    if precedients_triplets is None:
        with open("dataset/lsipcr-precs-anon-triplet.json", "r") as f:
            precedients_triplets = json.load(f)
    return precedients_triplets