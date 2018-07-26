# TextCNN

## BACKGROUND
- Identify the Rebate Websites 

## Requirements
- Python3
- tensorflow
- sklearn
- jieba
- scikit-learn

## Model



## WHAT I DO
- use the pre-trained Chinese word vector provided by Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, Analogical Reasoning on Chinese Morphological and Semantic Relations, ACL 2018.

- use Top 20 TD-IDF words as Features 

- slightly changes in codes

## FURTURE

- Top 20 tag words are used to classification so far, try to extract more words 
- Use title/description/keywords to enrich the feature, which would perform better in terms of representing website. Since a website is consist of sevral div sections, which contains texts, images and links. Due to This word focuses on text information, each section can be represented by different embeddings, and ensamble them togather as the feature map. But the potantial risk is the exponential computation consumption, which lowering the effecicency. It is a tradeoff to balance the complexcity and efficiency.

- RNN/LSTM 
- Due to the limit of pre-trained word-embedding dictionary, some words inevitably don't have the corrresponding word vectors. Therefore, radom vectors were assinged and the embedding layer is set to trainable in the CNN model. with the growth of label data, the personalized word vectors is necessary.

## REFERENCE
- Dennybritz's Blog  http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
- Pre-trained Chinese Word Vector https://github.com/Embedding/Chinese-Word-Vectors
- Bright Small's Post http://www.brightideasinanalytics.com/rnn-pretrained-word-vectors/

