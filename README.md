# Music-Genre-Classification-Model-Convolutional-Neural-Network

## Introduction

With the rising popularity of music streaming apps and emerging artists, there's a significant increase in the number of published songs. Services like Spotify and Apple Music could greatly benefit from automated and accurate music genre classification to enhance their library organization and user recommendations. Our project aims to develop a neural network, specifically a ResNet, for accurate music genre classification from audio content. The project includes collecting data, building and training a ResNet, optimizing its parameters, and evaluating its accuracy. Our team, passionate about music, believes that deep learning can enhance user experiences by offering better recommendations and managing large data volumes efficiently.

## Background and Related Work

Music genre recognition, a subset of Music Information Retrieval (MIR), involves complex and subjective processes. Past research has predominantly utilized MP3 and WAV audio files. Significant milestones include:

1. **2006**: Introduction of a “deep confidence network” for genre classification using SVM learning and beat spectrum analysis.
2. **2011**: Development of feature vectors based on pitch and rhythm, leading to an 81% accuracy.
3. **2019**: An 18-layer Residual Neural Network (ResNet) by Indian engineers, achieving 82 – 94.5% accuracy on the GZTAN dataset.
4. **2018**: A VGG-16 CNN model by a University of Waterloo student, reaching 89.1% accuracy on the Audio Set database.
5. **2002**: A study on defining music genres by common characteristics using statistical pattern recognition software, achieving 61% accuracy across 10 genres.

## Data Processing

Initially trained on the GTZAN dataset, our model faced overfitting issues due to the dataset's limited size. We expanded our dataset by integrating data from Spotify, using a script that extracted song genres and URLs for downloading. We collected about 10,000 songs, which were then preprocessed and converted into spectrograms for model training.

### Data Preprocessing Steps:
1. **Data Loading**: Songs were loaded and split into 30-second segments in .wav format.
2. **Data Conversion**: Conversion of audio files to spectrograms using Fourier transforms and the librosa library.
3. **Data Partitioning**: Splitting the data into training (60%), validation (20%), and testing (20%) sets.
4. **Tensor Conversion**: Converting images to tensors with dimensions 418x627x3 for model input.

## Architecture

Our primary model is a modified 18-layer ResNet, inspired by Microsoft Research's design. The model includes convolutional and fully connected layers, dropout techniques to prevent overfitting, and batch normalization for better training performance. We optimized the model through various hyperparameters, including epochs, learning rate, and batch size.

## Baseline Model

We compared our ResNet model against baseline models like K-Nearest Neighbour (KNN) and Artificial Neural Network (ANN) to assess its performance. The KNN model was set with a K value of 55, and the ANN featured two layers with ReLU activation functions.

## Quantitative Results

The models were evaluated based on their accuracy in matching the top prediction with the ground truth label. Our ResNet model showed superior performance in training, validation, and testing accuracies compared to the baseline models.

## Qualitative Results

The ResNet model showed higher accuracy in classifying genres like classical, jazz, and reggae. However, it had lower confidence in genres like rock, blues, and pop. The model's performance varied across different genres, indicating the challenge of identifying distinctive features in each genre.

## New Data

To evaluate the model's real-world performance, we tested it on new data from sources like YouTube. This included covers, genre-swapped songs, and live performances. The model's accuracy on this new data was lower than on the original test set but still outperformed the baseline models.

## Discussion

While our ResNet model outperformed the baseline models, it showed signs of overfitting. We attempted various techniques like dropout, data augmentation, and learning rate adjustments to optimize the model. The performance varied across genres, reflecting the challenge of capturing distinctive features in music spectrograms.

## Ethical Considerations

We considered informed consent, bias and fairness, and privacy in our data collection and model usage. The model's effectiveness can vary based on the music collection size and the quality of the audio source.

## Project Difficulty/Quality

The project was medium to high in difficulty, given the complexity of the ResNet model and the challenges in identifying distinct features in music spectrograms. We explored various data augmentation techniques and model structures to improve accuracy.

## Conclusion

Our ResNet model demonstrates the potential of neural networks in music genre classification, achieving a testing accuracy of 94.2%. The project highlights the challenges in dealing with diverse and complex music data. Future recommendations include exploring more color data augmentation techniques and considering advanced architectures like Vision Transformers (ViT) for improved performance.