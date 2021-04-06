### Annotation system of CKGC data set

As far as we know, existing KGC (Knowledge Grounded Conversation) datasets only focus on grounding conversations in a monolingual knowledge corpus. Thus, we annotate a CKGC (Cross-lingual Knowledge Grounded Conversation) test dataset . Below, we provide our dataset construction details.

With a largest number of articles in Wikipedia, English Wikipedia is applied as the unstructured knowledge base to ground the open-domain conversations in knowledge-scare languages. Following previous works, we consider a one-to-one conversation scenario in CKGC, and only one participant (i.e., the *wizard*) has access to an information retrieval system that shows the worker paragraphs from Wikipedia possibly relevant to the conversation, while the other is a curious learner (the *apprentice*)

Before the start of the conversation, two participants engage in chitchat will be randomly assigned a roles of *wizard* or *apprentice*, and the *apprentice* has to choose the topic of conversation. Then, the two participants chat one by one, while the *wizard* can access some knowledge which is unobservable to the *apprentice*. The conversation repeats until one of the conversation partners ends the chat.

#### Topic selection. 

Dinan et al. crowd-sourced a set of natural, open-domain dialogue topics for the Wizard-of-Wikipedia. Thus, we use topics that have appeared in unseen test set in Wizard of Wikipedia as topic sets in our dataset. Before the conversation, we show several topics randomly selected from the topic sets to the *apprentice*. The *apprentice* then chooses a topic of interest as the start topic of the conversation. During the conversation, the topic is allowed to naturally change.

#### Knowledge retrieval. 

During the conversation, the *wizard* has access to a set of passages of knowledge which may be relevant to the given dialogue context. There are two types of knowledge shown to the *wizard*, one is about original topic, and the other is the knowledge updated in real time as the conversation progresses. For the first type, since each topic linked to a Wikipedia article, we use the first 10 sentences in this article, which usually the summary of the article. For the second type, we retrieve the top 7 articles (first paragraph only) for the last two turns of dialogue to adapt to the topic transition in conversation. Specifically, we first translate each utterance in the conversation context using google translation. Then, we retrieve the articles via Apache Solr, a open source enterprise search platform built on Apache Lucene. We sort the knowledge based on the unigram F1 correlation of the dialogue context and each piece of knowledge.

#### Quality assurance. 

We hired 6 experienced experts to score each conversation collected in previous step. We ask experts to evaluate three aspects of the data, including knowledge relevance (whether the selected knowledge is relevant to the context), correctness of knowledge representation (whether the wizard understands and uses the knowledge correctly) and dialogue coherence (whether the two parties are engaged in the dialogue), and assign a score in 0, 1, 2 (representing “bad”, “fair”, and “good”). Each data is evaluated by two experts repeatedly to eliminate bias. We deleted all data scored as “bad” in any of the three aspects.