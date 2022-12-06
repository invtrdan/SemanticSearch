from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# corpus_dict = {
#     "yesList1" : ["I", "do", "with", "difficulty","yes","hard", "By", "myself","own"],
#     "yesList2" : ["yes","I","do","me","myself","alone", "by", "myself", "without", "can"],
#     "yesList3" : ["yes","do","with", "help","and","helps", "my", "somebody", "someone", "son", "daughter"],
#     "noList4" : ["no","somebody","does", "for", "me", "my","son","daughter","friend", "husband"],
#     "noList5" : ["never", "hard", "for", "me", "no", "did"],
#     "noList6" : ["next","other", "no", "easy", "don't", "do", "not","never", "myself", "without"]
#     }#!/usr/bin/python

# https://docs.google.com/document/d/12p9NxuRykKOwaKBEEGvrzHBoJtdeAyJ9kX98FDslBno/edit?usp=sharing
questions = [
  "Do you write checks, pay bills, and balancing checkbooks by yourself?",
  "Do you assemble tax records, business affairs, or papers by yourself?",
  "Do you go shopping alone for clothes, household necessities, or groceries?"
# "Do you play a game of skill or working on a hobby?"
# "Do you heat water, make a cup of coffee, turn off stove after use by yourself?"
# "Do you prepare a balanced meal 
# Keeping track of current events 
# Paying attention to, understanding, discussing TV, book, magazine 
# Remembering appointments, family occasions, holidays,  medications 
# Traveling out of neighborhood, driving, arranging to take buses 
# ]
  
]
#format- key=question number, key= [[categories of resposes for each question],[corresponding score for category]]
categories = {
    1: [[
    'Yes I write cheques and pay bills by myself',
    'I write cheques and pay bills by myself but it is challenging to do so',
    'I write checks and pay bills myself but I receive help from a person I know',
    'I am unable to write checks and pay bills, it is difficult for me to do so, my caregiver helps me',
    'I have never written cheques or paid bills but I am certain I can do so with ease',
    'I have never written cheques or paid bills but I am certain that it would be challenging for me to do so'
    ], [
    1,
    1,
    2,
    3,
    0,
    1,
    ]],
    2: [[
    'I do my taxes and prepare document by myself',
    'I do my taxes and prepare document by myself but it is challenging to do so',
    'I often get help from someone else when I am preparing my taxes or other documents',
    'I do not do taxes by myself, someone else doe it for me',
    'I have never done taxes or file document before but I am certain that it would be easy for me',
    'I have never done taxes or file document before but I am certain that it would be challenging'
    ], [
    1,
    1,
    2,
    3,
    0,
    1,
    ]],
    3: [[
    'I do all of my shopping by myself',
    'I do my shopping on my own but I have some challenges',
    'I do my shopping with the help of someone',
    'I do not do my own shopping, someone else does it for me',
    'I have never been shopping but I am certain that I can get it done easily',
    'I have never been shopping but I am certain that it would be challenging'
    ], [
    1,
    1,
    2,
    3,
    0,
    1,
    ]]
}

encoded_categories = dict()
for k, v in categories.items():
  encoded_categories[k] = [model.encode(v[0]), v[1]]
  
# model = SentenceTransformer('all-MiniLM-L6-v2') #pre-trained model
