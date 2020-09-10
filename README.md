## Tagging Danish texts using _Den Danske Begrebsordbog_

This repository contains code for a prototype semantic tagger for Danish language texts, using linguistic information taken from _[Den Danske Begrebsordbog](https://dsl.dk/projekter/den-danske-begrebsordbog)_.

The goal of this project is to be able to annotate Danish texts in a way that is compuationally light, conceptually simple, and linguistically intutive. This will allow researchers in Arts, Humanities, and Social Science subjects to extract semantic information from their corpora of Danish texts.

Possible research questions this tool can be used to address:

  - How does the distribution of a certain concept change over time?
  - How are different concepts represented in different corpora- newspapers vs parliamentary discourse, for example?
  - What associations can be found between genre/text type and the kinds of semantic categories which they contain?
  - Are there any significant associations between semantic categories? 
  - Etc.


## Key points

### Annotation data

Den Danske Begrebsordbog (_DBO_ for short) groups words together into 888 different semantic fields, an overview of which can be found _[here](https://www.dansksproghistorie.dk/wp-content/uploads/2016/08/19.-Den-Danske-Begrebsordbogs-kapitel-og-afsnitsoversigt.pdf)_. These semantic fields have a unique identifier, while some words belong to multiple categories. For example, consider the following entry:

```
    semantik | 11045505 | sb. | 12.012 Betydning | 13.18 Humaniora
```

Here we see a word (_semantik_); an ID number unique to this word (_11045505_); POS (_sb._). There are then two possible meanings for this word, taken from two different semantic fields - _12.012 Betydning_ or _13.18 Humaniora_. The tagger has two levels of granularity: a top-level category; and a sub-category. For example, _12.012 Betydning_ has the top-level category _12 Tegn, meddelelse, sprog_.

Note that the DBO structure includes Danish part-of-speech tags (e.g. _sb._ or _substantive_). As part of a preprocessing step, all of the POS tags in the DBO were converted to a Universal POS tagset ([link](https://universaldependencies.org/treebanks/da_ddt/index.html)).

### NLP Framework

At present, the tagger uses the _[UDPipe](http://ufal.mff.cuni.cz/udpipe)_ framework to annotate texts. There are a number of pragmatic reasons for choosing this framework over others (e.g. stanfordNLP, spaCy, OpenNLP on the JVM). Ultimately, UDPipe won out for the breadth of it's linguistic annotations, it's speed, and the fact that it ships with an easy-to-use REST server implementation (see _Issues and Further Work_ below).

### Disambiguating Polysemous Words

Disambigutation of polysemous word senses is an open problem in NLP, particularly for low resource languages such as Danish. To try to address this, the tagger uses Jaccard Distance as a method of infering the most likely category. This distance is calculated using the set of all words in context window ±5 words, as well as their parts-of-speech. The smaller the Jaccard Distance between the set of context words and the individual sets of possible categories for a target word, the more likely it is to be the correct sense of that word.

The disambiguatiom algorithm has the following steps:

  - For each word, take all the other words in a window of ±5 words, along with their POS tag.
  - For the target word, find all possible categories from the DBO hierarchy.
  - For each possible category, calculate the Jaccard Distance between the set of word_POS in the context and the set of all word_POS in the top level category for each tag.
  - Return an ordered list of three most likely tags based on lowest Jaccard Distance.
  
The algorithm uses the top-level category for disambigutation, rather than the sub-category. This is based on a linguistic intuition regarding the distribution of semantic categories. For any given word, it is unlikely that there will be many context words which belong to the exact same sub-category. However, sub-categories which belong to the same top-level category share a degree of semantic similarity. 

Take for example, _13.005 Geometri, figur_ and _13.004 Matematik_. Both are clearly related and in fact both belong to the top-level category _13 Videnskab_. It thus seems likely to assume that the presence of these tags would likely be accompanied by more words from _13 Videnskab_, rather than from the specfic sub-categories. This allows the algorithm to infer tags based on a looser set of semantic relations.
  

## How to run the prototype

(NB: For copyright reasons, the data from DBO cannot be included in this repository. These instructions assume you have the data, in a folder called _dict_)

First, create a virtual environment to work in. Then you should activate the virtual environemnt and install the necessary requirements.

For example:

```
python3 -m venv env
source ./env/bin/activate
pip3 install -r requirements.txt
```

Next, save any data you want to tag as plain text (.txt) files in the folder called _*in*_. Then simply run the script from the root directory:

```
python3 src/DNO_tag.py
```

If things are working correctly, you should get some feedback in the terminal about the progress of the script. 

And that's it! Your output files are saved in folder called _*out*_.


## Issues and Future Work

At the moment, this requires Python ≥ 3.6 and only works on macOS and Linux. 

From a linguistic perspective, there needs to be some evaluation of the accuracy of the tagger. Further word sense disambiguation methods can be explored.

From a technical perspective, the tagger should be refactored in order to be used as an module that can be imported into other programs.

Lastly, version of this software is currently being developed using Rust. This will cross compile to other operating systems and will require no runtime dependencies. Watch this space.


## Data and results

_[Det Danske Sprog- og Litteraturselskab](https://dsl.dk/)_ have kindly allowed me access to the data behind _Den Danske Begrebsordbog_ for the purposes of developing this prototype.

However, due to copyright issues, this repo contains _only_ the code used for the tagger, along with example input and output files. It does not contain any data related to the contents of DBO.

For more information, please contact the author directly.

## Author

Author:   [rdkm89](https://github.com/rdkm89) <br>
Date:     September 2020

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

"Ordbogsdata fra Den Danske Begrebsordbog, © Det Danske Sprog- og Litteraturselskab
2020".
