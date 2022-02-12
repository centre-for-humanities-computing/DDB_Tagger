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

import os, sys, argparse, glob
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
            self.nlp = spacy.load("da_core_news_sm") # could be changed

        elif da_model == "dacy":
            import dacy
            self.nlp = dacy.load("medium") # could be changed

    def tag_text(self, input:str, input_file: bool=False, only_tagged_results: bool=False):
        """Processing and tagging a text using Den Danske Begrebsordbog.

        Args:
            input (str): String of input text or path to input file. 
            input_file (bool, optional): Defines whether input is path (True) to input file or string of text (False). Defaults to False.
            only_tagged_results (bool, optional): Defines whether results should only contain tags (True), or also scores (False). Defaults to False.
        """   

        # --- PREPARE TEXT ---     

        # If input is file, load file
        if input_file == True:
            with open(input, 'r') as f:
                text = f.read() 
                f.close()

        # If input is string, use it as text
        elif input_file == False:
            text = input

        # --- TOKENIZE AND POS TAGGING --- 

        # Tokenize and save with POS tags in tuple (what happens with nan POS tags?)
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
        # Save index as column
        token_pos.insert(loc=0, column='ORIGINAL_IDX', value=token_pos.index)
        # Turning dataframe into dictionaries (with keys ORIGINAL_IDX, TOKEN, POS)
        token_dicts = token_pos.to_dict("records")
        # Converting to a list of dicts without punctuation
        dicts_no_punc = [token_d for token_d in token_dicts if token_d["POS"] !="PUNCT"]
        # Also saving dataframe with punctuation, to later concatenate again
        dicts_punc = [dict(token_d, **{'DDB_TAGS': "-", "DDB_TAGS_DISAMBIGUATED": "-"}) for token_d in token_dicts if token_d["POS"] == "PUNCT"]
        
        # --- INITIAL TAGGING ---

        tagged = []
        for idx, token_d in enumerate(dicts_no_punc):

            # Prepare target token
            target = token_d["TOKEN"].lower() + "_" + token_d["POS"]
            # Find DDB tags in which the target occurs
            target_tags = [tag for tag, tag_tokens in self.DDB_dict.items() if target in tag_tokens]
            
            # If token does not appear in any category, append "-"
            if len(target_tags) == 0:
                target_tagged_scores = "-"

            # Otherwise calculate scores for categories
            else: 
                # Get the context tokens of the target
                context_dicts = self.get_context(dicts_no_punc, idx)
                context = [token_d["TOKEN"].lower() + "_" + token_d["POS"] for token_d in context_dicts]

                # Calculate scores for possible categories based on context
                target_tags_scores = []
                for tag in target_tags: 
                    # Get the top level tag
                    top_level_tag = tag.split("|")[0] + "|"
                    # Get all the tokens of the category
                    top_level_tokens = list(chain.from_iterable([tag_tokens for tag, tag_tokens in self.DDB_dict.items() if top_level_tag in tag]))
                    # Calculate distance of context words and category tokens
                    score = jaccard_distance(context, top_level_tokens) 
                    # Append tuple of category and score
                    target_tags_scores.append((tag, score))
                    # Sort possibel tags by score
                    target_tags_scores = sorted(target_tags_scores, key=lambda x: x[1])

            # Add results to dict 
            target_tagged_dict = dict(token_d, **{'DDB_TAGS': target_tags_scores})
            # Append result for token to list of all tagged tokens
            tagged.append(target_tagged_dict)

        # --- DISAMBIGUATION OF IDENTICAL SCORES ---

        # Prepare file to save information about disambiguation
        filepath = "out/disambiguation_info.txt"
        if os.path.exists(filepath):
            os.remove(filepath)

        # Loop through tagged tokens
        tagged_disambiguated = []
        for idx, token_tagged_d in enumerate(tagged):
            
            # Get the possible tags for the given token
            tags_scores = token_tagged_d["DDB_TAGS"]
            # Get only the scores of the possible tags
            scores = [ts[1] for ts in tags_scores if tags_scores != "-"]
            # Get duplicate scores (if there are any in the first 4 scores)
            duplicate_scores = list(set([s for s in scores[:4] if scores[:4].count(s) > 1]))

            # If there are any duplicate scores
            if len(duplicate_scores) >= 1:

                # Get the idx and the tags of those with duplicate scores
                duplicate_idxs = [tags_scores.index(ts) for ts in tags_scores if ts[1] == duplicate_scores[0]]
                duplicate_tags_scores = [tags_scores[idx] for idx in duplicate_idxs]

                # Get the dictionaries of the context tokens
                context_tagged = self.get_context(tagged, idx)
                # Get the tags of the context tokens (only if they should not be disambiguated themselves)
                tags_context = []
                for token in context_tagged:
                    # Get the tags of the token
                    tags_token = token["DDB_TAGS"]
                    # Add the top1 tag, but only if it should not also be disambiguated
                    if (len(tags_token) == 1 and tags_token != "-") or (len(tags_token) > 1 and tags_token[0][1] != tags_token[1][1]):
                        # Add the tag of the first level tag to the tags_context list
                        tags_context.append(token["DDB_TAGS"][0][0])

                # Disambiguate the tags of the target based on the context (or size of the tag entry in DDB)
                duplicates_disambiguated = self.disambiguate_duplicates(token_tagged_d, duplicate_tags_scores, tags_context, filepath)
                # Fix order of the duplicates, together with the non-duplicates
                tags_disambiguated = tags_scores.copy()
                for idx, duplicate in enumerate(duplicates_disambiguated):
                    new_idx = duplicate_idxs[idx]
                    tags_disambiguated[new_idx] = duplicate
            
            # If no duplicates, the ordered is just the same as the original
            else:
                tags_disambiguated = tags_scores

            # Save results (top 3 tags) in a copy of the dictionary, to prevent overwriting and append
            token_disambiguated_d = token_tagged_d.copy()
            token_disambiguated_d["DDB_TAGS"] = tags_scores[:3]
            token_disambiguated_d["DDB_TAGS_DISAMBIGUATED"] = tags_disambiguated[:3]
            tagged_disambiguated.append(token_disambiguated_d)

        # --- PREPARE OUTPUT ---

        # Put tagged punct and no punct into dataframes and join
        column_names = ['ORIGINAL_IDX', 'TOKEN', 'POS', 'DDB_TAGS', 'DDB_TAGS_DISAMBIGUATED']
        df_tagged = pd.DataFrame(tagged_disambiguated, columns=column_names) 
        df_punc = pd.DataFrame(dicts_punc, columns=column_names)
        output = pd.concat([df_tagged, df_punc]).sort_values("ORIGINAL_IDX").reset_index(drop=True)
        
        # Split top three categories, drops unnecessary rows and replace NAs
        output[["DDB1", "DDB2", "DDB3"]] = pd.DataFrame(output["DDB_TAGS_DISAMBIGUATED"].values.tolist())
        output = output.drop(["DDB_TAGS", "DDB_TAGS_DISAMBIGUATED"], axis=1)
        output = output.fillna("-")

        # If scores should not be in input only get the score
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

    def get_context(self, input_list:list, target_idx:int):
        """Helper function to get 5 entries before and after a target in list.

        Args:
            input_list (list): List of elements, from which to extract context of a target.
            target_idx (int): Index of the target

        Returns:
            context (list): List of context entries from list
        """

        # For the first 5 words
        if target_idx <= 5:
            pre_target = input_list[:target_idx]
            post_target = input_list[target_idx+1:target_idx+6]

        # Last 5 words
        elif target_idx >= len(input_list)-5:
            pre_target = input_list[target_idx-5:target_idx]
            post_target = input_list[target_idx:]

        # All other words
        else: 
            pre_target = input_list[target_idx-5:target_idx]
            post_target = input_list[target_idx+1:target_idx+6]

        # Join context before and after
        context = pre_target + post_target

        return context
    
    def disambiguate_duplicates(self, token_tagged_d:dict, duplicate_tags_scores:list[tuple], tags_context:list[str], filepath:str):
        """Helper function to disambiguate between tags with duplicate scores.

        Args:
            token_tagged_d (dict): dictionary of target token, for which scores are duplicate (only for info file)
            duplicate_tags_scores (list[tuple]): list of tuples (tag, score) of tags with duplicate scores
            tags_context (list[str]): list of str for the tags from the context
            filepath (str): filepath to save disambiguation info

        Returns:
            duplicates_disambiguated (list[tuple]): list of tuples disambiguated by necessary sorting algorithm
        """

        # --- SAVE BASIC INFORMATION ---
        
        f = open(filepath, "a")
        f.write("\n----------------------------------------------\n")
        f.write(f"TARGET INFO: {token_tagged_d['ORIGINAL_IDX'], token_tagged_d['TOKEN']}\n")
        f.write(f"TARGET ALL TAGS: {token_tagged_d['DDB_TAGS']}\n")
        f.write(f"TARGET DUPLICATE TAGS: {duplicate_tags_scores}\n")
        f.write(f"TAGS CONTEXT: {tags_context}\n")
            
        # --- HIGH LEVEL DISAMBIGUATION ---

        # Get high level tags of the context
        top_tags_context = [tag.split("|")[0] + "|" for tag in tags_context]

        # Count how many context words have each of the possible tags
        top_tags_counts = [(tag, top_tags_context.count(tag[0].split("|")[0] + "|")) for tag in duplicate_tags_scores]
        top_counts = [tag_count[1] for tag_count in top_tags_counts]
        f.write(f"TOP LEVEL TAG COUNTS: {top_tags_counts}\n")
        
        # If that was successful in disambiguating
        if len(top_counts) == len(set(top_counts)): 
            top_tags_counts_ordered = sorted(top_tags_counts, key=lambda x: x[1], reverse=True)
            duplicates_disambiguated = [tag[0] for tag in top_tags_counts_ordered]

        # --- LOW LEVEL DISAMBIGUATION ---
        
        else: 
            # Count how many context words have each of the possible tags
            sub_tags_counts = [(tag, tags_context.count(tag[0])) for tag in duplicate_tags_scores]
            sub_counts = [tag_count[1] for tag_count in sub_tags_counts]
            f.write(f"SUB LEVEL TAGS COUNTS: {sub_counts}\n")
            
            # If that was successful in disambiguating
            if len(sub_counts) == len(set(sub_counts)):
                sub_tags_counts_ordered = sorted(sub_tags_counts, key=lambda x: x[1], reverse=True)
                duplicates_disambiguated = [tag[0] for tag in sub_tags_counts_ordered]


        # --- CATEGORY SIZE DISAMBIGUATION ---

            else:
                # Return sorted by size of low level category
                f.write(f"TAG SIZE DISAMBIGUATION: {[(x, len(self.DDB_dict[x[0]])) for x in duplicate_tags_scores]}\n")
                duplicates_disambiguated = sorted(duplicate_tags_scores, key=lambda x: len(self.DDB_dict[x[0]]), reverse=True)

        return duplicates_disambiguated


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

    