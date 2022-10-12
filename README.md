# natural-language-questions


## Documentation (WIP)
 - The documentation is more technical description of what you have done and how and why as well as any data supporting your results and conclusions.
 

 ### Introduction
 Natural Language Question Answering is an up-and-coming field with many real-world applications like surfacing better results from search queries. As there are many models from Long Short-Term Memory models and Transformers that are used in this space, this project seeks to consolidate exisiting literature on such models in the context of HotpotQA. HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. It is collected by a team of NLP researchers at Carnegie Mellon University, Stanford University, and Université de Montréal.

 The goal of this project will be to experiment, summarise the strengths and weaknesses of existing models for Question and Answering machine learning tasks. 

 ### Introduction to HotPotQA Question Answering Dataset
 In this section, I will introduce the dataset. There are three HotpotQA files:
 - Training set http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
 - Dev set in the distractor setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
 - Dev set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json This is just hotpot_dev_distractor_v1.json without the gold paragraphs, but instead with the top 10 paragraphs obtained using our retrieval system.
 - Test set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json The context in the file is paragraphs obtained using the author's retrieval system, which might or might not contain the gold paragraphs.

 A sample of the data is as shown below:
 '''
 {
  "supporting_facts": [
    [
      "Arthur's Magazine",
      0
    ],
    [
      "First for Women",
      0
    ]
  ],
  "level": "medium",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "context": [
    [
      "Radio City (Indian radio station)",
      [
        "Radio City is India's first private FM radio station and was started on 3 July 2001.",
        ...
      ]
    ],
    [
      "History of Albanian football",
      [
        "Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
        ...
      ]
    ]
  ],
  "answer": "Arthur's Magazine",
  "_id": "5a7a06935542990198eaf050",
  "type": "comparison"
  }
'''

The top level structure of each JSON file is a list, where each entry represents a question-answer data point. Each data point is a dict with the following keys:

- _id: a unique id for this question-answer data point. This is useful for evaluation.
question: a string.
- answer: a string. The test set does not have this key.
- supporting_facts: a list. Each entry in the list is a list with two elements [title, sent_id], where title denotes the title of the paragraph, and sent_id denotes the supporting fact's id (0-based) in this paragraph. The test set does not have this key.
- context: a list. Each entry is a paragraph, which is represented as a list with two elements [title, sentences] and sentences is a list of strings.

There are other keys that are not used in our code, but might be used for other purposes (note that these keys are not present in the test sets, and your model should not rely on these two keys for making preditions on the test sets):

- type: either comparison or bridge, indicating the question type. (See our paper for more details).
- level: one of easy, medium, and hard. (See paper for more details).


 ### DistilBERT
 #### Motivation
 According to Hugging Face:
 "DistilBERT [is] a distilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.".

 Hence, DistilBERT served as a good benchmark for future experiments in terms of the accuracy and training durations.
 #### Results
 As the 








