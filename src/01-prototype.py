#!/usr/bin/env python3

"""
Import libraries to be used.
"""
import time
import os
import pandas as pd
from ufal.udpipe import Model, Pipeline, ProcessingError

"""
Import super_dict
"""
print("Loading DBO data...")
# load tagging data
df = pd.read_csv("dict/tagging-map-POS.csv", sep="\t", encoding="utf-8")
# Create dict/hashmap
hashmap = dict(zip(df.WORD, df.DBO_TAG))
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


"""
Tagging
"""
# Counter for number of files tagged
tagged_no = 0
#Timer
start = time.time()

# Main loop
for filename in os.listdir("in/"):
    if not filename.startswith('.'):
        """
        Read in file
        TODO: glob.
        """

        print(f"Working on {filename}...")
        infile = "in/" + filename
        outfile = "out/" + filename
        with open(infile, 'r') as f:
            text = f.read()
            f.close()

        """
        Tokenize and POS tag using stanfordNLP
        """

        print("Tokenizing and POS tagging...")
        # create doc object
        data = pd.DataFrame([y.split("\t") for y in pipeline.process(text,error).split("\n")]).dropna().set_index(0).reset_index(drop=True)
        # create list of (word,pos) tuples from doc object
        tups = list(zip(data[1], data[3]))
        print("...done!")


        """
        stanfordNLP has separate POS tags for coordianting and subordinating conjunctions, particles, and auxiliaries.
        This is absent in the DBO data, so we need to convert all the Stanford UPOS tags to a single KONJ representation
        """
        print("Cleaning up POS tags...")
        cleaned = []
        for k,v in tups:
            if k == "at":
                cleaned.append((k,"PART"))
            elif v == "CCONJ" or v == "SCONJ":
                cleaned.append((k,"KONJ"))
            elif v == "AUX":
                cleaned.append((k,"VERB"))
            else:
                cleaned.append((k,v))
        print("...done!")


        """
        For word,pos tuple, return all matching categories.
        Note that, at this stage, there is no sense disambiguation.
        """
        print("Tagging with DBO categories...")
        tagged = []
        for word,pos in cleaned:
            lowered = word.lower()
            target = lowered+"_"+pos
            dbo_tags = hashmap.get(target, "--")
            tagged.append((word, pos, dbo_tags))
        print("...done!")

        """
        Create DataFrame and save
        """
        print("Saving results...")
        df = pd.DataFrame(tagged)
        df.columns = ["WORD", "POS", "DBO_TAG"]
        df.to_csv(outfile, index=False, sep="\t", encoding="utf-8")
        print("...done!")
        print("\n ================== \n")
        tagged_no += 1


"""
Print info for user
"""
print(f"{tagged_no} files tagged in {round(time.time()-start, 2)} seconds!")
