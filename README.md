### Project Description:

This dataset consists of reviews of fine foods from amazon downloaded from Kaggle. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.   

[Data Link](https://www.kaggle.com/snap/amazon-fine-food-reviews)

The dataset includes text review and a short summary for the same and we will use this information and this information was used to train the model.
  
The model architecture involved Encoder-Decoder structure where Encoder consists of 3-layers stacked LSTM and Decoder consists of single layer LSTM.

The attention mechanism takes output from encoder and decoder and tries to learn the important parts of input for generating the output/summary. Basically, attention mechanism provides a method to weigh the input words to understand their importance in predicting the next word in summary.


#### Python Libraries Utilized in the project:
- Numpy, Pandas: For data preperation, wrangling, cleaning and manipulation
- pickle: Storing data in compressed binary format
- re: Python library to take advantage of regular expressions for data manipulation
- BeautifulSoup (bs4): For extracting information from html tags
- NLTK: For natural language processing
- Scikit-Learn: For dividing data into train-test split
- Matplotlib: For visualizing data and results
- Keras, Tensorflow: For preparing neural network structure and training model
