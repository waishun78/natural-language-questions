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
 ```
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
```

The top level structure of each JSON file is a list, where each entry represents a question-answer data point. Each data point is a dict with the following keys:

- _id: a unique id for this question-answer data point. This is useful for evaluation.
question: a string.
- answer: a string. The test set does not have this key.
- supporting_facts: a list. Each entry in the list is a list with two elements [title, sent_id], where title denotes the title of the paragraph, and sent_id denotes the supporting fact's id (0-based) in this paragraph. The test set does not have this key.
- context: a list. Each entry is a paragraph, which is represented as a list with two elements [title, sentences] and sentences is a list of strings.

There are other keys that are not used in our code, but might be used for other purposes (note that these keys are not present in the test sets, and your model should not rely on these two keys for making preditions on the test sets):

- type: either comparison or bridge, indicating the question type. (See our paper for more details).
- level: one of easy, medium, and hard. (See paper for more details).

For the purpose of this project, I will be using
hotpot_train_v1.1 : http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
hotpot_dev_distractor_v1 : http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
datasets. The hotpot_train_v1.1 and hotpot_dev_distractor_v1 will be referred to train_set.json and dev_set.json respectively in subsequent sections. The former being the dataset used to train the model and the latter being the validation dataset.

 ### DistilBERT
 #### Motivation
 According to Hugging Face:
 "DistilBERT [is] a distilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.".

 Hence, DistilBERT served as a good benchmark for future experiments in terms of the accuracy and training durations.
 
 #### Preprocessing
 To feed the texts to the model, a tokenizer (corresponding to the desired question and answering model used) is used to tokenize the inputs. This converts the words to the corresponding token IDs in the pretrained vocabulary and format it for the model to use. It also generates other features like the attention mask, start_positions and end_positions.
 ```train_encodings``` and ```val_encodings``` has keys:
 - input_ids: tokenized values of the question and context for each dataset
 - attention_mask: indicate what corresponding vales in the input_ids should be attended to and what should not be (has the same size as input_ids)
 - start_positions: the index of the value in the input_ids at which the answer starts
 - end_positions: the index of the value in the input_ids at which the answer ends
 
 The data set did not provide the start index and end index which would be used to do training. Hence, we had to manually search for the index of the answers in the context. Since the context was a nested list, it is  first flattened before the index was found using the ```.find()``` function. If the answer is not found in the original example, such as in the case of yes/no answers, the question was ignored.
 
 To load the dataset to the DistilBertForQuestionAnswering model, I initalised a custom Dataset object.
 
 To fine-tune the pre-trained 'distilbert-base-uncased' model, I loaded the pre-trained model into the model. Using the AdamW optimiser on the model parameters, I trained the entire data set for n epochs. To load the dataset to the model to train, I used a dataloader that can specify the b batch_size for the training.
 
 The hyperparameters batch_size and epochs are potential parameters that can be changed to obtain a better model.
 
 Using the loss obtained at each batch, I calculated the loss and used that to optimise the model parameters.
 
 The model is then evaluated by using accuracy and precision (recall was calculated but not referenced) and compared with other models to find the efficacy of the model.
 
 #### Results
 
Example of answers produced by the model:
| Number      | Question | Answer     | Predicted     |
| :---:        |    :----:   |          :---: |    :----:   |
| 1.| What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? | Chief of Protocol | lord high treasurer or lord treasurer was an english government position and has been a british government position since the acts of union of 1707. a holder of the post would be the third - highest - ranked great officer of state, below the lord high steward and the lord high chancellor. a kiss for corlissa kiss for corliss is a 1949 american comedy film directed by richard wallace and written by howard dimsdale. it stars shirley temple in her final starring role as well as her final film appearance. it is a sequel to the 1945 film " kiss and tell ". " a kiss for corliss " was retitled " almost a bride " before release and this title appears in the title sequence |
| 2.| What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species? | animorphs | animorphs " series, written by k. a. applegate. with respect to continuity within the series, it takes place before book # 23, " the pretender ", although the events told in the story occur between the time of " the ellimist chronicles " and " the andalite chronicles ". the book is introduced by tobias, who flies to the valley of the free hork - bajir, where jara hamee tells him the story of how the yeerks enslaved the hork - bajir, and how aldrea, an andalite, and her companion, dak hamee, a hork - bajir, tried to save their world from the invasion. jara |
| 3.| The director of the romantic comedy "Big Stone Gap" is based in what New York city? | greenwich village, new york city | greenwich village, new york city. trigiani has published a novel a year since 2000. great eastern conventionsgreat eastern conventions, inc. was an entertainment company which produced comic book conventions, most actively during the years 1987 - 1996. in new york city, the great eastern shows filled the gap between the mid - 1980s demise of the annual comic art convention and creation conventions, and the establishment of promoter michael carbonaro's annual big apple comic con in 1996. from 1993 – 1995, great eastern hosted two new york city shows annually at the jacob k. javits convention center. great eastern also ran shows in new jersey, pennsylvania, massachusetts, oregon, minnesota, and texas. new york society of model engineersthe new york society of model engineers ( nysme ) was originally incorporated in 1926 in new york city. there are published records that show the society existed as early as 1905. in its early years, the organization moved to and from various locations throughout manhattan. at that time it was basically a gentlemen's club of members who were interested in all types of model building. in 1926 the society was formalized and incorporated under the laws of the state of new york. this |

 Due to time constraints and other limitations elaborated in the later section, I was only able to conduct fine-tuning with the number of training epochs and the batch size use for training.
| Epoch      | Batch Size | Accuracy     | Precision     |
| :---:        |    :----:   |          :---: |    :----:   |
| 5      |    16     |   0.350   |   0.00202    |
| 7      |    16     |   0.345   |   0.0148    |
| 10      |    16     |   0.353   |   0.00497    |
| 32      |    8     |   0.359   |   0.00346    |

As the number of parameters needed to train the distilbert question and answering model is too large to run on my local computer (2.6 GHz 6-Core Intel Core i7, 16 GB RAM) as well as Google Colab Pro, I was reliant on external servers. 

Accuracy and precision were the two primary evaluation criterion used. This is because accuracy can artificially inflate the efficacy of the model. As we see, the accuracy of all the models were significantly better than precision. This is likely due to the fact that the model would often predict a longer answer to make sure that the answer is captured in it. Precision accounts for that by penalising false positives/unnecessarily long answers).

From our results, we observed that the combination of 7 epochs and batch size of 16 achieved a significantly higher precision but a slight deprovement on accuracy as compared to other experiments. Further analysis would be needed to ensure its validity and to understand the reason for this significant improvement in precision.

However, from qualitative results, we can see that the answers predicted by the model is often much longer than the actual answer. This indicates a skew towards the longer answers. A potential improvement is to penalise unnecessarily long answers more heavily.

 #### Limitations 
 In light of the limited computing power available for the project, the following are a few measures that were taken to reduce the RAM usage for training while ensuring that the training time remains reasonable.
 
 1. Experimenting with the batch size. 
  - Batch size refers to the number of samples that are used to train a model before updating the training model variables. With a smaller RAM available, the amount of data the computer can accumulate will be lower. This creates an upper bound for the maximum batch size we are able to run on the system. 
2. Increasing the GPU usage.
- Using ```device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')# move model over to detected device ``` while using Google Colab. However, even then, the kernel would crash prematurely before training is completed.
3. Experimenting gradient accumulation
- By accumulating gradient, we do not need to calculate the gradients of the entire batch but instead do it iteratively in smaller batches, only running the optimisation step when enough gradients are accumulating.
- This allows for a "larger batch size" without exceeding the machine's memory limit.
 
 #### Future Improvements
 What other things you wish to experiement on? 
 - Tokenisation without consideration truncation of the question.
 - Other models available

 

 
 
 





