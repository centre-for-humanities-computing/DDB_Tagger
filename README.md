## Tagging Danish texts using _Den Danske Begrebsordbog_

This repository contains code for a prototype semantic tagger for Danish language texts, using linguistic information taken from _[Den Danske Begrebsordbog](https://dsl.dk/projekter/den-danske-begrebsordbog)_.

The goal of this project is to be able to annotate Danish texts in a way that is compuationally light, conceptually simple, and linguistically plausible. This will allow researchers in Arts, Humanities, and Social Science subjects to extract semantic information from their corpora of Danish texts.

Possible research questions this tool can be used to address:

  - How does the distribution of a certain concept change over time?
  - How are different concepts represented in different corpora- newspapers vs parliamentary discourse, for example?
  - What associations can be found between genre/text type and the kinds of semantic categories which they contain?
  - Are there any significant associations between semantic categories? 
  - Etc.


## Key points

### Annotation data

Den Danske Begrebsordbog (_DBO_ for short) groups words together into 888 different semantic fields. These semantic fields have a unique identifier, while some words belong to multiple categories. For example, consider the following entry:

```
    semantik | 11045505 | sb. | 12.012 Betydning | 13.18 Humaniora
```

Here we see a word (_semantik_); an ID number unique to this word; POS (_sb._). There are then two possible meanings for this word, taken from two different semantic fields - _12.012 Betydning_ or _13.18 Humaniora_.

### NLP Framework

At present, the tagger uses the _[UDPipe](http://ufal.mff.cuni.cz/udpipe)_ framework to annotate texts. There are a number of pragmatic reasons for choosing this framework over others (e.g. stanfordNLP, spaCy, OpenNLP on the JVM). Ultimately, UDPipe won out for the breadth of it's linguistic annotations, it's speed, and the fact that it ships with an easy-to-use REST server implementation (see _Issues and Further Work_ below).

### Polysemous words

For the current prototype, the tagger returns all possible semantic fields associated with a word. These are ranked relative to which sense is most 'central' for the word in question. To do this, I used a TFIDF vectorizer to find the 'keywords' for each category. 

If a high TFIDF score indicates keyness, then it stands to reason that a low TFIDF score suggests the sense is more central to the semantics of that word. Thus, I assume that the lowest TFIDF score for each category tends to correspond to the root or most basic sense of that word.

The tagger therefore currently returns a tab seperated file, with the word, Part-of-Speech, and all possible DBO tags, ranked according the following formula:

```
  category_rank = 1 - TFIDF
```

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
python3 src/01-prototype.py
```

If things are working correctly, you should get some feedback in the terminal about the progress of the script. 

And that's it! Your output files are saved in folder called _*out*_.


## Issues and Future Work

At the moment, this requires Python ≥ 3.6 and only works on macOS and Linux. 

From a linguistic perspective, the current tagger is quite unsophisticated. It does not perform any contextual word-sense disambiguation, instead returning categories based on a measure of centrality. Future iterations will explore more detailed disambiguation methods, in order to improve the accuracy.

A version of this software is currently being developed using Rust. This will cross compile to other operating systems and will require no runtime dependencies. Watch this space.


## Data and results

_[Det Danske Sprog- og Litteraturselskab](https://dsl.dk/)_ have kindly allowed me access to the data behind _Den Danske Begrebsordbog_ for the purposes of developing this prototype.

However, due to copyright issues, this repo contains _only_ the code used for the tagger, along with example input and output files. It does not contain any data related to the contents of DBO.

For more information, please contact the author directly.

## Author

Author:   [rdkm89](https://github.com/rdkm89) <br>
Date:     March 2020

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

"Ordbogsdata fra Den Danske Begrebsordbog, © Det Danske Sprog- og Litteraturselskab
2020".
