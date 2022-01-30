"""
Semantic Tagger using Danske Begrebsordbog.
This script can be used in three ways:

1 Import DDB_Tagger Class in a notebook to tag a text:
    - from DDB_Tagger import DDB_Tagger # import class
    - Tagger = DDB_Tagger(dict="dict/dict.pkl, da_model="spacy") # initialise tagger
    - results = Tagger.tag_text(input="this is your text", input_file=False, only_tagged_results=False)

2 Import DDB_Tagger Class in a notebook to all texts in a given directory:
    - from DDB_Tagger import DDB_Tagger # import class
    - Tagger = DDB_Tagger(dict="dict/dict.pkl, da_model="spacy") # initialise tagger
    - results = Tagger.tag_directory(input_path="in/", output_path="out/", only_tagged_results=False)
 
2 Run the script to tag all texts in a given dictionary:
    - from DDB_Tagger directory run from Terminal: python3 src/DDB_Tagger.py --input_directory "in/" --output_directory out/ --da_model "spacy" --dict "dict/dict.pkl"
    - Results will be saved in --output_directory

Both ways require that:
    - Requirements are installed install_requirements.sh
    - DDB dictionary is stored as pickle file in dict/dict.pkl
"""

# --- DEPENDENCIES ---

import sys, argparse, glob
from tqdm import tqdm
import time
import pickle
import pandas as pd
from itertools import chain


# --- HELPER FUNCTIONS ---

def jaccard_distance(context_tokens: list[str], category_tokens: list[str]):
    """Simple function for calculating Jaccard Distance

    Args:
        context_tokens (list[str]): List of context words (i.e. target words)
        category_tokens (list[str]): List of words in category

    Returns:
        [float]: Value of jaccard distance

    Info:

        s1: The set of all words+POS ±5 from target word
        s2: The set of all words+POS from the top-level DDB category

        JD = (s1 ∪ s2) - (s1 ∩ s2) / (s1 ∪ s2)

        This can also be calculated using 1 - (s1 ∩ s2) / (s1 ∪ s2), where the 
        latter expression is the Jaccard similarity.
    """

    s1 = set(context_tokens)
    s2 = set(category_tokens)
    union = len(s1.union(s2))
    intersection = len(s1.intersection(s2))
    return (union-intersection) / union


# --- DDB TAGGER CLASS ---

class DDB_tagger:

    def __init__(self, dict: str="dict/dict.pkl", da_model: str="spacy"):
        """Initializing Semantic Tagger using Den Danske Begrebsordbog.

        Args:
            dict (str, optional): Path to semantic dictionary. Defaults to "dict/dict.pkl".
            da_model (str, optional): Danish Language Model to use, "spacy" or "dacy". Defaults to "spacy".
        """        

        # Load DDB dictionary
        self.DDB_dict = pickle.load(open(dict, "rb"))

        # Load Danish language model
        if da_model == "spacy":
            import spacy
            self.nlp = spacy.load("da_core_news_sm")

        elif da_model == "dacy":
            import dacy
            self.nlp = dacy.load("medium") # could be changed

    def tag_text(self, input: str, input_file: bool=False, only_tagged_results: bool=False):
        """Processing and tagging a text using Den Danske Begrebsordbog.

        Args:
            input (str): String of input text or path to input file. 
            input_file (bool, optional): Defines whether input is path (True) to input file or string of text (False). Defaults to False.
            only_tagged_results (bool, optional): Defines whether results should only contain tags (True), or also scores (False). Defaults to False.
        """   

        # --- PREPARE TEXT ---     

        if input_file == True:
            with open(input, 'r') as f:
                text = f.read() 
                f.close()

        elif input_file == False:
            text = input

        # --- TOKENIZING AND POS TAGGING ---

        # Tokenize and save with POS tags in tuple (nan POS tags?)
        token_pos = pd.DataFrame([(token.text, token.pos_) for token in self.nlp(text)], columns = ["TOKEN", "POS"])
        # Remove rows with NAN
        token_pos.dropna(inplace=True) 
        # Remove rows with "SPACE" tag
        token_pos = token_pos[token_pos["POS"] != "SPACE"]
        # Rename POS tag for "at", since it is only defined as PART in DDB
        token_pos.loc[token_pos['TOKEN'] == 'at', 'POS'] = 'KONJ'
        # Rename POS tags to match the DDB tags (i.e. converting universal POS to DDB POS)
        token_pos["POS"].replace({"CCONJ": "KONJ", 
                                  "SCONJ": "KONJ",
                                  "AUX": "VERB"}, inplace=True)

        # Reset index to account for deleted SPACE tags
        token_pos = token_pos.reset_index(drop=True)
        # Turning dataframe into tuples
        tuples = token_pos.to_records(index=True)
        # Converting to a list of tuples with punctuation
        tuples_no_punc = [(original_idx, token, pos) for (original_idx, token, pos) in tuples if pos!="PUNCT"]
        tuples_punc = [(original_idx, token, pos, "-") for (original_idx, token, pos) in tuples if pos=="PUNCT"]

        # --- TAGGING ---

        tagged = []
        for idx, (original_idx, token, pos) in enumerate(tuples_no_punc):

            # PREPARE WORD 
            target = token.lower() + "_" + pos

            # PREPARE CONTEXT 
            # First 5 words
            if idx <= 5:
                pre_target = tuples_no_punc[:idx]
                post_target = tuples_no_punc[idx+1:idx+6]

            # Last 5 words
            elif idx >= len(tuples_no_punc)-5:
                pre_target = tuples_no_punc[idx-5:idx]
                post_target = tuples_no_punc[idx:]

            # All other words
            else: 
                pre_target = tuples_no_punc[idx-5:idx]
                post_target = tuples_no_punc[idx+1:idx+6]

            # Prepare context words with POS tags
            context = [token.lower() + "_" + pos for (_, token, pos) in pre_target + post_target]

            # FIND CATEGORIES CONTAINING THE TARGET
            categories = [category for category, category_tokens in self.DDB_dict.items() if target in category_tokens]
            # sort? : keys.sort(key=lambda x: x[1], reverse=True)
            
            # If token does not appear in any category, append -
            if len(categories) == 0:
                tagged.append((original_idx, token, pos, "-"))

            # Otherwise calculate scores for category
            else: 
                scores = []
                for cat in categories: 
                    top_level_tag = cat.split("|")[0] + "|"
                    top_level_tokens = list(chain.from_iterable([category_tokens for category, category_tokens in self.DDB_dict.items() if top_level_tag in category]))
                    score = jaccard_distance(context, top_level_tokens) 
                    scores.append((cat, score))
                    top_results = sorted(scores, key=lambda token: token[1])[:3]
                tagged.append([original_idx,token,pos,top_results])

        # --- PROCESS OUTPUT ---

        # Put tagged punct and no punct into dataframes
        df_tagged = pd.DataFrame(tagged, columns=['INDEX', 'WORD', 'POS', 'DDB_TAG'])
        df_punc = pd.DataFrame(tuples_punc, columns = ['INDEX', 'WORD', 'POS', 'DDB_TAG'])
        # Join punct and no punct dataframes
        output = pd.concat([df_tagged, df_punc]).sort_values("INDEX").reset_index(drop=True)
        # Split top three categories
        output[["DDB1", "DDB2", "DDB3"]] = pd.DataFrame(output["DDB_TAG"].values.tolist())
        # Drop DDB_TAG column
        del output["DDB_TAG"]
        # Replace "-" with None
        output = output.fillna("-")

        # If scores should not be in input
        if only_tagged_results == True:
            output["DDB1"] = output["DDB1"].str[0]
            output["DDB2"] = output["DDB2"].str[0]
            output["DDB3"] = output["DDB3"].str[0]
        
        # Return output
        return output

    def tag_directory(self, input_directory: str="in/", output_directory: str="out/", only_tagged_results: bool=False):
        """Tagging all texts (.txt files) in a directory using the tag_text function.

        Args:
            input_directory (str, optional): Input directory containing .txt files. Defaults to "in/".
            output_directory (str, optional): Output directory to save results. Defaults to "out/".
            only_tagged_results (bool, optional): Defines whether results should only contain tags (True), or also scores (False). Defaults to False.
        """        
        
        # --- PREPARING FILENAMES ---

        # Get filenames of directory
        file_pattern = input_directory + "*.txt"
        filenames = glob.glob(file_pattern)

        # If no files found:
        if len(filenames) == 0:
            sys.exit(f"[ERROR] No files matching {file_pattern} found, check input_directory path or file placement.")
        else:
            print(f"[INFO] Found {len(filenames)} files, starting tagging...")

        # --- TAGGING ALL FILES AND SAVING OUTPUTS ---

        # Start timer 
        start = time.time()

        # Loop tagger over files
        for filename in tqdm(filenames):
            output = self.tag_text(input=filename, input_file=True, only_tagged_results=only_tagged_results)
            output_filename = output_directory + ("only_" if only_tagged_results == True else "scores_") + "tagged_" + filename.split("/")[1].replace(".txt", ".csv")
            output.to_csv(output_filename, index=False, sep="\t", encoding="utf-8")
        
        # Print done and results
        print(f"[INFO] ...done! Results saved in {output_directory}")

        # Print timings
        print(f"[INFO] {len(filenames)} files tagged in {round(time.time()-start, 2)} seconds!")
        print("\n ================== \n")



# --- RUN TAGGER FOR FILES IN DIRECTORY ---

if __name__ == '__main__':

    # --- REQUIREMENT: PYTHON >= 3.6 ----
    if sys.version_info[:2] < (3,6):
        sys.exit("[ERROR] Oops! You need Python 3.6+!")
    print("\n ================== \n")

    # --- ARGUMENT PARSER ---
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_directory', type=str, required=False,
                        default="in/",
                        help='Input directory to files to tag, will tag all files in directory.')

    parser.add_argument('--dict', type=str, required=False,
                        default="dict/dict.pkl",
                        help="Path to semantic dictionary.")

    parser.add_argument('--da_model', type=str, required=False,
                        default="spacy",
                        help="Danish language model to use, 'spacy' or 'dacy'.")

    parser.add_argument('--only_tagged_results', required=False,
                        action="store_true", default=False,
                        help="Use argument if results should only contain, by default also contain scores.")

    parser.add_argument('--output_directory', type=str, required=False,
                        default="out/",
                        help="Directory to save output files.")

    args = parser.parse_args()
    
    # -- RUN TAGGER FOR DIRECTORY ---

    # Loading tagger (add error message if da_model is not loaded?)
    Tagger = DDB_tagger(dict=args.dict, da_model=args.da_model)
    print(f"[INFO] DDB Tagger with {args.da_model} loaded, now processing files...")

    # Run tagger for directory 
    Tagger.tag_directory(input_directory=args.input_directory, 
                         output_directory=args.output_directory, 
                         only_tagged_results=args.only_tagged_results)

    