"""
Danielle McIntosh
Semantic Search
"""

from sentence_transformers import util
import random
import torch
from corpora import questions, categories, encoded_categories, model

def semantic_search(query, qid):
    #sentences to be embedded
    score_list = []
    
    top_k = 5
    
    #embedding sentences using model.encode()
    corpus_embeddings = encoded_categories[qid+1][0]
    
    #query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
     
    #using cosine-similarity and torch.topk to find the corpus embedding with the highest score
    highest_score = util.cos_sim(query_embedding, corpus_embeddings)[0]
    embedding = torch.topk(highest_score, k=top_k)
    
    print("\n\n============================\n\n")
    print("Query:", query)
    print("\n5 most similar sentences to categories of question: ", qid+1)

    results = {}
    for score, idx in zip(embedding[0], embedding[1]):
      # print(idx)
      print(categories[qid+1][0][idx], "(Score: {:.4f})".format(score))
      results[qid+1] = categories[qid+1][1][idx]
        # score_list.append([corpus[idx], score])
        
    # max_score = score_list[0][1]
    # entry = score[0][0]
    # for pair in score_list:
    #     if pair[1] > max_score:
    #         max_score = pair[1]
    #         entry = pair[0]
        
    return results


question_bank = [
  "Do you write checks, pay bills, and balancing checkbooks by yourself?",
  "Do you assemble tax records, business affairs, or papers by yourself?",
  "Do you go shopping alone for clothes, household necessities, or groceries?",
  "Do you play a game of skill or working on a hobby by yourself?",
  "Do you heat water, make a cup of coffeeor turn off stove after use by yourself?",
  "Do you prepare a balanced meal by yourself?",
  "Do you keep track of current events by yourself?",
  "Do you pay attention to, understand or discuss TV, books or magazine by yourself?",
  "Do you remember appointments, family occasions, holidays, or time to take medications by yourself?",
  "Do you travel outside of your neighborhood by driving or taking the bus by yourself?"
]

categories_ = {
  1: [[
    'I write cheques and pay bills by myself.',
    'I write cheques and pay bills by myself but it is challenging to do so.',
    'I write checks and pay bills myself, but I receive help from a person I know.',
    'I am unable to write checks and pay bills, it is difficult for me to do so, my caregiver helps me.',
    'I have never written cheques or paid bills, but I am certain I can do so with ease.',
    'I have never written cheques or paid bills, but I am certain that I would be challenging for me to do so.'
  ], [1, 1, 2, 3, 0, 1]],
  2: [[
    'I do my taxes and prepare documents by myself.',
    'I do my taxes and prepare documents by myself, but it is challenging to do so.',
    'I often get help from someone else when I am preparing my taxes or other documents.',
    'I do not do taxes by myself, someone else does it for me.',
    'I have never done taxes or filed documents before, but I am certain that it would be easy for me.',
    'I have never done taxes or filed documents before, but I am certain that it would be challenging.'
  ], [1, 1, 2, 3, 0, 1]],
  3: [[
    'I do all my shopping by myself.',
    'I do my shopping on my own, but I have some challenges.',
    'I do my shopping with the help of someone.',
    'I do not do my own shopping, someone else does it for me.',
    'I have never been shopping but I am certain that I can get it done easily.',
    'I have never been shopping but I am certain that it would be challenging.'
  ], [1, 1, 2, 3, 0, 1]],
  4: [[
    'I play games of skill or work on a hobby.',
    'I play games of skill or work on a hobby, but I have challenges doing so.',
    'I play games of skill or work on a hobby with help from someone.',
    'I do not play games of skill or work on a hobby.',
    'I have never played games of skill or work on a hobby, but I am certain that I can get it done easily',
    'I have never played games of skill or work on a hobby, but I am certain that it would be challenging'
  ], [1, 1, 2, 3, 0, 1]],
  5: [[
    'I prepare tea by myself.', 'I have a hard time making a cup of coffe.',
    'Someone has to help me whenever I am making tea.',
    'I do not make tea for myself, someone prepares it for me.',
    'I have never made tea for myself, but I am certain I can do so easily.',
    'I have never made tea for myself, but I am certain it would be difficult for me to do so.'
  ], [1, 1, 2, 3, 0, 1]],
  6: [[
    'I prepare a balanced meal by myself.',
    'I prepare a balanced meal by myself but it is challenging for me to do so.',
    'I normally get help from someone when I am cooking.',
    'I do not cook, someone else does the cooking.',
    'I do not cook however I am certain I can cook easily.',
    'I do not cook but I am certain it would be challenging for me to do so.'
  ], [1, 1, 2, 3, 0, 1]],
  7: [[
    'I keep track of current events by myself.',
    'It is challenging for me to keep track of current events by myself.',
    'I keep track of current events with help from someone.',
    'I do not keep tracks of current events, someone does it for me.',
    'I do not keep track of current events, but I can easily do so.',
    'I do not keep track of current events, but it would be challenging for me to do so.'
  ], [1, 1, 2, 3, 0, 1]],
  8: [[
    'I watch tv and or read books by myself.',
    'I watch tv and read books but it is challenging for me to do so.',
    'I watch tv and read books with help from someone.',
    'I do not watch tv or read books someone relays the information to me.',
    'I do not watch tv or read books, but I can do so with ease.,',
    'I do not watch tv or read books, but it would be challenging for me to do so,'
  ], [1, 1, 2, 3, 0, 1]],
  9: [[
    'I remember family occasions and appointments by myself.',
    'I find it hard to remember family occasions and appointments by myself.',
    'I remember family occasions and appointments with help from someone.',
    'Someone else reminds me of family occasions and appointments.',
    'I do not remember family occasions or appointments, but I can easily remember them if I want to.',
    'I do not pay attention to family occasions or appointments, but it would be hard for me to do so.'
  ], [1, 1, 2, 3, 0, 1]],
  10: [[
    'I often travel on my own.', 'I travel on my own, but it is difficult',
    'I often travel with someone else.', 'I do not leave my home.',
    'I do not leave my home, but I am certain I can do so.',
    'I do not leave my home, but I am certain it will be challenging for me.'
  ], [1, 1, 2, 3, 0, 1]]
}


NUM_QUESTIONS = 10
question_id = random.randint(0, NUM_QUESTIONS)

print('Question #:', question_id+1)
print('\n\nAnswer this question:')
print(question_bank[question_id])
user_response = input("\n\nEnter Answer: ")

question_score = semantic_search(user_response, question_id)

print('\nScoring for this question..')
print('Question Number: Score from question\n')
print(question_score)







#Question One - Writing checks, paying bills, balancing checkbook
#categories: ['I write cheques and pay bills by myself', 'I write cheques and pay bills by myself but it is challenging to do so', 'I write checks and pay bills myself but I receive help from someone I know','I am unable to write checks and pay bills, it is difficult for me to do so, my son does it for me', 'I have never written cheques or paid bills but I am certain I can do so with ease', 'I have never written cheques or paid bills but I am certain that it would be challenging for me to do so']

#Question Two - Assembling tax records, business affairs, or papers
#categories: ['I do my taxes and prepare document by myself', 'I do my taxes and prepare document by myself but it is challenging to do so', 'I often get help from someone else when I am preparing my taxes or other documents','I do not do taxes by myself, someone else doe it for me', 'I have never done taxes or file document before but I am certain that it would be easy for me', 'I have never done taxes or file document before but I am certain that it would be challenging']

#Question Three - Shopping alone for clothes, household necessities, or groceries
#categories: ['I do all of my shopping by myself', 'I do my shopping on my own but I have some challenges','I do my shopping with the help of someone','I do not do my own shopping, someone else does it for me','I have never been shopping but I am certain that I can get it done easily', 'I have never been shopping but I am certain that it would be challenging']

#QuestionFour - Playing a game of skill, working on a hobby
#Categories: ['I normally play golf or knit in my free time', 'I play golf or knit in my free time but it is challenging to do so', 'I play golf with help from someone','I do not play any games or hobbies in my free time','I do not play games or hobbies in my free time but I am certain that I can easily do so', 'I do not play games or hobbies in my free time but I am certain that it would be challenging to do so']

#QuestionFive- Heating water, making a cup of coffee, turning off stove after use
#Categories: ['I prepare tea by myself','I have a hard time making a cup of coffe','Someone has to help; me whenever I am making tea','I do not make tea for myself, someone prepares it for me ','I have never made tea for myself but I am certain I can do so easily','I have never made tea for myself but I am certain it would be difficult for me to do so']

#QuestionSix - Preparing a balanced meal
#Categories: ['I prepare a balanced meal by myself','I prepare a balanced meal by myself but it is challenging for me to do so', 'I normally get help from someone when I am cooking', 'I do not cook someone else does the cooking','I do not cook however I am certain I can cook easily','I do not cook but I am certain it would be challenging for me to do so']

#QuestionSeven - Keeping track of current events
#Categories: ['I keep track of current event by myself','It is challenging for me to keep track of current events by myself','I keep track of current events with help from someone','I do not keep tracks of current events, someone does it for me', 'I do not keep track of current events but I can easily do so', 'I do not keep track of current events but it would be challenging for me to do so']

#QuestionEight - Paying attention to, understanding, discussing TV, book, magazine
#Categories: ['I watch tv and or read books by myself','I watch tv and read books but it is challenging for me to do so','I watch tv and read books with help from someone','I do not watch tv or read books someone relays the information to me','I do not watch tv or read books but I can do so with ease', 'I do not watch tv or read books but it would be challenging for me to do so']

#QuestionNine - Remembering appointments, family occasions, holidays,  medications
#Categories: ['I remember family occasions and appointments by myself',' I find it hard to remember family occasions and appointmments by myslef','I remember famuly occasions and appointments with help from someone,'I do not remember family occasions or appointments but I can easily rememeber them if I want to','I do not pay attention to family occasions or appointments but it would be hard for me to do so']

#QuestionTen - Traveling out of neighborhood, driving, arranging to take buses
#Categories: ['I often travel on my own','I travel on my own but it is difficult','I often travel with someone else','I do not leave my home','I do not leave my home but I am certain I can do so','I do not leave my home but I am certain it would be challenging for me']