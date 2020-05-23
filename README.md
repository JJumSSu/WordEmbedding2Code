# Word Embedding Compression
Pytorch implementation of the paper

*Compressing Word Embeddings Via Deep Compositional Code Learning* (Shu et al., ICLR 2018)  [[link]](https://arxiv.org/abs/1711.01068)

## Dependencies

* python 3.6
* pytorch (>1.0)
* numpy
* keras (for downloading IMDB Dataset)

## How to use

### Download Source Code

Clone.

```
git clone https://github.com/JJumSSu/WordEmbedding2Code
```

Move to directory containing source files.

```
cd WE_to_Code/src
```

### Get GloVe

```
bash get_glove.sh
```

### Train Code Learner

Specify the name of the model output.

The glove embedding file will be spliitted into train and valid datasets.

```
make train MODEL_NAME=Your_Compressor_Model_Name M=32 K=16
```

* Please note that the loss explodes occasionally even though the seed is fixed.

  When the 'NaN' pops up during training(or a very large number), please re-execute the command.
  
  Normally it should work like as in the figure below.

<p align="center">
  <img  src=screenshot.PNG>
</p>
  

### Evaluate Code Learner

Specify the name of the trained model(compressing).

Evluation will be conducted on the valid dataset.

```
make evluate MODEL_NAME=Your_Compressor_Model_Name
```

### Run Sentiment Analysis Task

Specify the name of the trained compressing model and name of the classifier.

Evluation will be conducted on the official test dataset(IMDB).

Using original GloVe

```
make run_classifier_glove MODEL_NAME=Output_Model_Name GLOVE_MODEL_NAME=Your_Compressor_Model_Name
```

Using compressed GloVe

```
make run_classifier_glove_compressed MODEL_NAME=Output_Model_Name GLOVE_MODEL_NAME=Your_Compressor_Model_Name
```

## Result

### Reconstruction Loss 

Results evaluated on the hedlout validation dataset(42B.300d)

|Method|Eculidian|Euclidian^2|
|------|---|---|
|With Codes|4.276|18.280
|With Logits|3.492|12.192

### Sentiment Classification Accuracy

Results evaluated on the IMDB test dataset

|Method|Accuracy|
|------|---|
|With Original GloVe|0.881|
|With Compressed GloVe|0.870|

