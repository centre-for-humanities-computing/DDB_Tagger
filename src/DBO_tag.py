#!/usr/bin/env python3
#TODO: Glob files; refactor for efficiency; better solution than mp.pool?

"""
Import libraries to be used.
"""
import sys, time, os
import pickle
from itertools import chain
import tqdm
import multiprocessing as mp
import pandas as pd
from ufal.udpipe import Model, Pipeline, ProcessingError

def jaccard_distance(target, DBO_category):
    """
    Simple function for calculating Jaccard Distance

    s1: The set of all words+POS ±5 from target word
    s2: The set of all words+POS from the top-level DBO category

    JD = (s1 ∪ s2) - (s1 ∩ s2) / (s1 ∪ s2)

    This can also be calculated using 1 - (s1 ∩ s2) / (s1 ∪ s2), where the 
    latter expression is the Jaccard similarity.
    """

    s1 = set(target)
    s2 = set(DBO_category)
    union = len(s1.union(s2))
    intersection = len(s1.intersection(s2))
    return (union-intersection) / union

# Main function
def tagfiles(filename) :
    """
    Main function to tag texts with DBO categories
    """
    # Create I/O names
    infile = "in/" + filename
    outfile = "out/" + filename

    # Read in file (TODO: glob).
    with open(infile, 'r') as f:
        text = f.read()
        f.close()

    # Tokenize and POS tag using UDPipe
    # create doc object
    data = pd.DataFrame([y.split("\t") for y in pipeline.process(text,error).split("\n")])\
            .dropna()\
            .set_index(0)\
            .reset_index(drop=True)
    # create list of (word,pos) tuples from doc object
    tups = list(zip(data[1], data[3]))


    """
    UdPipe has separate POS tags for:
        - coordinating and subordinating conjunctions,
        - particles,
        - and auxiliaries.

    This is absent in the DBO data, so we need to convert all UPOS tags.
    """
    cleaned = []
    for idx,(k,v) in enumerate(tups):
        if k == "at":
            cleaned.append((idx,(k,"PART")))
        elif v == "CCONJ" or v == "SCONJ":
            cleaned.append((idx,(k,"KONJ")))
        elif v == "AUX":
            cleaned.append((idx,(k,"VERB")))
        else:
            cleaned.append((idx,(k,v)))


    """
    For each (word,pos) tuple, return all matching categories.
    
    Sense disambiguation is performed by calculating the Jaccard Disance between the context words
    and the top level category for the target word.
    """

    # Remove text tagged as punctuation; keep in indexed list for later
    no_punc = [(idx,(word,pos)) for (idx,(word,pos)) in cleaned if pos!="PUNCT"]
    punc = [[idx,word,pos,"--"] for (idx,(word,pos)) in cleaned if pos=="PUNCT"]

    # Empty list of DBO tagged text to go in
    tagged = []
    for idx,tup in enumerate(no_punc):
        # For first five words
        if idx<=5:
            # Extract the context around the target word
            context = [cword.lower()+"_"+cpos for cidx,(cword,cpos) in no_punc[:idx]+no_punc[idx+1:idx+6]]
            # Target word
            word = tup[1][0]
            # Target POS
            pos = tup[1][1]
            # Join word_POS
            target = word.lower()+"_"+pos
            # Find all categories which contain the target
            if [(target,key) for key,value in super_dict.items() if target in value]:
                keys = [key for key,value in super_dict.items() if target in value]
                keys.sort(key=lambda x: x[1], reverse=True)
                scores = []
                # For each of those categories, take only the top level tag
                for CAT in keys:
                    tag = CAT.split('|')[0]+"|"
                    # Calculate jaccard using context and all words in all sub-categories of top level
                    full_set = list(chain.from_iterable([value for key, value in super_dict.items() if tag in key]))
                    score = jaccard_distance(context,full_set)
                    scores.append((CAT, score))
                top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                tagged.append((idx,word,pos,top_results))
            else:
                tagged.append((idx,word,pos,"--"))

        # From sixth word to the fifth from the end; otherwise logic identical to above
        elif idx > 5 and idx <= (len(no_punc)-5):
            # Extract the context around the target word
            context = [cword.lower()+"_"+cpos for cidx,(cword,cpos) in no_punc[idx-5:idx]+no_punc[idx+1:idx+6]]
            # Target word
            word = tup[1][0]
            # Target POS
            pos = tup[1][1]
            # Join word_POS
            target = word.lower()+"_"+pos
            # Find all categories which contain the target
            if [(target,key) for key,value in super_dict.items() if target in value]:
                keys = [key for key,value in super_dict.items() if target in value]
                keys.sort(key=lambda x: x[1], reverse=True)
                scores = []
                # For each of those categories, take only the top level tag
                for CAT in keys:
                    tag = CAT.split('|')[0]+"|"
                    # Calculate jaccard using context and all words in all sub-categories of top level
                    full_set = list(chain.from_iterable([value for key, value in super_dict.items() if tag in key]))
                    score = jaccard_distance(context,full_set)
                    scores.append((CAT, score))
                top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                tagged.append((idx,word,pos,top_results))
            else:
                tagged.append((idx,word,pos,"--"))

        # Last five words to end of text; otherwise logic identical to above
        elif idx >= len(no_punc)-5:
            # Extract the context around the target word
            context = [cword.lower()+"_"+cpos for cidx,(cword,cpos) in no_punc[idx-5:idx]+no_punc[idx:]]
            # Target word
            word = tup[1][0]
            # Target POS
            pos = tup[1][1]
            # Join word_POS
            target = word.lower()+"_"+pos
            # Find all categories which contain the target
            if [(target,key) for key,value in super_dict.items() if target in value]:
                keys = [key for key,value in super_dict.items() if target in value]
                keys.sort(key=lambda x: x[1], reverse=True)
                scores = []
                # For each of those categories, take only the top level tag
                for CAT in keys:
                    tag = CAT.split('|')[0]+"|"
                     # Calculate jaccard using context and all words in all sub-categories of top level
                    full_set = list(chain.from_iterable([value for key, value in super_dict.items() if tag in key]))
                    score = jaccard_distance(context,full_set)
                    scores.append((CAT, score))
                top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                tagged.append((idx,word,pos,top_results))
            else:
                tagged.append((idx,word,pos,"--"))

    """
    Create DataFrame and save
    """
    # Create df with column names
    df = pd.DataFrame(tagged)
    df.columns = ['INDEX', 'WORD', 'POS', 'DBO_TAG']
    # Joing DBO_TAG column as string for readability
    df['DBO_TAG'] = [','.join(map(str, l)) for l in df['DBO_TAG']]
    df['DBO_TAG'] = df['DBO_TAG'].replace('-,-' ,'--')

    """
    We need to update the indices on the removed punctuation, so that they fit into the new dataframe
    """
    # Update index and rename XPUNC for sorting. (No other POS has a tag beginning X or later)
    update = 0
    for item in punc:
        item[0] = item[0]-update
        item[2] = "XPUNCT"
        update+=1

    # Append punctutation to df on index
    df_with_punctuation = df.append(pd.DataFrame(punc, columns=df.columns))
    # Sort values on INDEX ascending and POS descending; ensures punctuation comes in the right place
    df_with_punctuation = df_with_punctuation.sort_values(['INDEX','POS'], ascending=[True, False])\
                                             .reset_index(drop=True)
    # Save to file
    df_with_punctuation.to_csv(outfile, index=False, sep="\t", encoding="utf-8")


if __name__ == '__main__':

    # Must be python 3.6 or higher!
    if sys.version_info[:2] < (3,6):
        sys.exit("Oops! You need Python 3.6+!")
    print("\n ================== \n")

    """
    Import super_dict

    This is the main source of DBO data for tagging. Dictionary structured the following way:

 
    { DBO Category 1: [list of words+POS in category],
      DBO Category 2: [list of words+POS in category],
      [...]
      DBO Category 888: [list of words+POS in category]
      }

    For each word, the POS tag in the super_dict comes from the DBO itself.

    """
    print("Loading DBO data...")
    # load tagging data
    super_dict = pickle.load(open("dict/dict.pkl", "rb"))
    print("...done!")
    print("\n ================== \n")


    """
    Import UDPipe Model

    This could be rewritten so that the tagging is done by a UDPipe REST server
    """
    print("Loading UDPipe model...")
    # load model
    model = Model.load("model/danish-ddt-ud-2.5-191206.udpipe")
    # initialise pipeline
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    error = ProcessingError()
    print("...done!")
    print("\n ================== \n")

    # Run tagger
    texts = []
    for filename in os.listdir("in"):
        if filename.endswith(".txt"):
            texts.append(filename)

    """
    Tag texts
    """
    print("Tagging texts...")
    # Start timer
    start = time.time()
    with mp.Pool(mp.cpu_count()-1) as p:
        list(tqdm.tqdm(p.imap(tagfiles, texts), total=len(texts)))
    p.close()
    p.join()
    print("...done!")
    print("\n ================== \n")
    # Print timings
    print(f"{len(texts)} files tagged in {round(time.time()-start, 2)} seconds!")
