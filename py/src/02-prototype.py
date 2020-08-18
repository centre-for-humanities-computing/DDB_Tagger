#!/usr/bin/env python3

"""
Import libraries to be used.
"""
import sys
import time
import os
import pickle
import pandas as pd
from itertools import chain
from ufal.udpipe import Model, Pipeline, ProcessingError 

def jaccard_distance(target, DBO_category):
    """
    Simple function for calculating Jaccard Distance
    """
    
    s1 = set(target)
    s2 = set(DBO_category)
    union = len(s1.union(s2))
    intersection = len(s1.intersection(s2))
    return (union-intersection) / union

# Main function
def tagfiles() :

    """
    Tagging
    """
    # Counter for number of files tagged
    tagged_no = 0
    #Timer
    start = time.time()

    # Iterate over all files in diretory, skipping noise like .DS_Store
    for filename in os.listdir("in/"):
        if not filename.startswith('.'):

            """
            Read in file (TODO: glob).
            """
            print(f"Working on {filename}...")
            infile = "in/" + filename
            outfile = "out/" + filename
            with open(infile, 'r') as f:
                text = f.read()
                f.close()

            """
            Tokenize and POS tag using UDPipe
            """
            print("Tokenizing and POS tagging...")
            # create doc object
            data = pd.DataFrame([y.split("\t") for y in pipeline.process(text,error).split("\n")])\
                    .dropna()\
                    .set_index(0)\
                    .reset_index(drop=True)
            # create list of (word,pos) tuples from doc object
            tups = list(zip(data[1], data[3]))
            print("...done!")


            """
            UdPipe has separate POS tags for:
                - coordinating and subordinating conjunctions,
                - particles,
                - and auxiliaries.

            This is absent in the DBO data, so we need to convert all UPOS tags.
            """
            print("Cleaning up POS tags...")
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
            print("...done!")
            

            """
            For word,pos tuple, return all matching categories.
            Note that, at this stage, there is no sense disambiguation.
            """
            print("Tagging with DBO categories...")

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
                            # Calculate jaccard using context words and all words in all categories comprising the top level category
                            score = jaccard_distance(context, list(chain.from_iterable([value for key, value in super_dict.items() if tag in key])))
                            scores.append((CAT, score))
                        top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                        tagged.append((idx,word,pos,top_results))
                    else:
                        tagged.append((idx,word,pos,"--"))
                
                # For everything from the sixth word to the fifth from the end; otherwise logic identical to above
                elif idx > 5 and idx <= (len(no_punc)-5):
                    context = [cword.lower()+"_"+cpos for cidx,(cword,cpos) in no_punc[idx-5:idx]+no_punc[idx+1:idx+6]]
                    word = tup[1][0]
                    pos = tup[1][1]
                    target = word.lower()+"_"+pos
                    if [(target,key) for key,value in super_dict.items() if target in value]:
                        keys = [key for key,value in super_dict.items() if target in value]
                        keys.sort(key=lambda x: x[1], reverse=True)
                        scores = []
                        for CAT in keys:
                            tag = CAT.split('|')[0]+"|"
                            score = jaccard_distance(context, list(chain.from_iterable([value for key, value in super_dict.items() if tag in key])))
                            scores.append((CAT, score))
                        top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                        tagged.append((idx,word,pos,top_results))
                    else:
                        tagged.append((idx,word,pos,"--"))
                
                # For the last five words of the text; otherwise logic identical to above
                elif idx >= len(no_punc)-5:
                    context = [cword.lower()+"_"+cpos for cidx,(cword,cpos) in no_punc[idx-5:idx]+no_punc[idx:]]
                    word = tup[1][0]
                    pos = tup[1][1]
                    target = word.lower()+"_"+pos
                    if [(target,key) for key,value in super_dict.items() if target in value]:
                        keys = [key for key,value in super_dict.items() if target in value]
                        keys.sort(key=lambda x: x[1], reverse=True)
                        scores = []
                        for CAT in keys:
                            tag = CAT.split('|')[0]+"|"
                            score = jaccard_distance(context, list(chain.from_iterable([value for key, value in super_dict.items() if tag in key])))
                            scores.append((CAT, score))
                        top_results = sorted(scores, key=lambda tup: tup[1])[:3]
                        tagged.append((idx,word,pos,top_results))
                    else:
                        tagged.append((idx,word,pos,"--"))
                        
            print("...done!")

            """
            Create DataFrame and save
            """
            print("Saving results...")
            # Create df with column names
            df = pd.DataFrame(tagged)
            df.columns = ['INDEX', 'WORD', 'POS", 'DBO_TAG']
            # Joing DBO_TAG column as string for readability
            df['DBO_TAG'] = [','.join(map(str, l)) for l in df['DBO_TAG']]
            df['DBO_TAG'] = df['DBO_TAG'].replace('-,-' ,'--')
            
            # Update punctuation index and rename XPUNC for sorting 
            origin = 0
            for item in punc:
                item[0] = item[0]-origin
                item[2] = "XPUNCT"
                origin+=1
            
            # Append punctutation to df
            df_with_punctuation = df.append(pd.DataFrame(punc, columns=df.columns))
            # Sort values on INDEX ascending and POS descending. This ensures that the punctuation comes in the right place
            df_with_punctuation = df_with_punctuation.sort_values(['INDEX','POS'], ascending=[True, False]).reset_index(drop=True)
            # Save to file
            df_with_punctuation.to_csv(outfile, index=False, sep="\t", encoding="utf-8")
            print("...done!")
            print("\n ================== \n")
            tagged_no += 1


    """
    Print info for user
    """
    print(f"{tagged_no} files tagged in {round(time.time()-start, 2)} seconds!")



if __name__ == '__main__':
    if sys.version_info[:2] < (3,6):
        sys.exit("Oops! You need Python 3.6+!")
    """
    Import super_dict
    """
    print("Loading DBO data...")
    # load tagging data
    # load pickled dic
    super_dict = pickle.load(open("dict/preprocessing/05-pickle-dict.pkl", "rb"))
    print("...done!")
    print("\n ================== \n")


    """
    Import UDPipe Model
    """
    print("Loading UDPipe model...")
    # load model
    model = Model.load("model/danish-ud-2.0-170801.udpipe")
    # initialise pipeline
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    error = ProcessingError()
    print("...done!")
    print("\n ================== \n")

    # Run tagger
    tagfiles()
