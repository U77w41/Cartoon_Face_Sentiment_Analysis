# Problem Statement 2: Detecting Emotional Sentiment in Cartoons


## Challenge Description:

Social media platforms are widely used by individuals and organizations to express emotions, opinions, and ideas. These platforms generate vast amounts of data, which can be analyzed to gain insights into user behavior, preferences, and sentiment. Accurately classifying the sentiment of social media posts can provide valuable insights for businesses, individuals, and organizations to make informed decisions.

To accomplish this task, a customized private cartoon dataset (original images) of social media posts has been provided, which contains labels for each post's emotion category, such as happy, angry, sad, or neutral.

The task is to build and fine-tune a machine-learning model that accurately classifies social media posts into their corresponding emotion categories, using synthetic images.

To achieve this, the following steps are required:
star Generate synthetic images using any image generation techniques (e.g., GAN, Diffusion Models, Autoencoder Decoder) to augment the dataset and increase its size.
star So for example, we use the images in the category of "happy" to synthetically generate similar images. Repeat the same for each category.
star Use the original and synthetic images to build a machine-learning model that accurately classifies social media posts into their corresponding emotion categories.
star Evaluate the performance of the model using appropriate metrics such as accuracy, precision, recall, and F1-score.
star Compare the performance of the model when trained on the original dataset only, the synthetic dataset only, and the combination of both.
star Analyze the results to determine the effectiveness of using synthetic images for improving classification accuracy.

The dataset consists of a diverse range of cropped cartoon face images. The data has been pre-processed and cleaned, but you can apply additional data cleaning or pre-processing techniques if necessary. You can use any machine learning or deep learning algorithm or technique of your choice to build and finetune your model, as long as it can accurately classify the posts into their corresponding emotion categories.

Based on a previous study, The performance accuracy of the best classification algorithms for emotion detection is 0.906. Your goal is to beat this using your models, but your model should not be overfitting or underfitting.

To evaluate the performance of your model, you will be using standard evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix. The submission with the highest evaluation score will be declared the winner. The top submissions will also be invited to present their solutions and insights to the community.



## My Solution Methodology:

**Step 1: Data Collection**

Dataset is provided by the competition host and can be found in [Kaggle](https://www.kaggle.com/datasets/revelation2k23/brain-dead-emotion-detection)

**Step 2: Data Preprocessing**

Preprocess the dataset to ensure that all images are of the same size and format.
Normalize the pixel values of the images to be between 0 and 1.

**Step 3: Train StyleGAN**

Train the StyleGAN model on the preprocessed dataset to learn the underlying
structure of the images. Then fine-tuned the model to improve the quality of the generated
images.

**Step 4: Generate Images**

Use the trained StyleGAN model to generate new images that are similar in style and
content to the original dataset.

**Step 5: Data Upsampling for Image Classification**

The images of the existing dataset are upsampled with the synthetic images
generated by the StyleGAN model.

**Step 6: Data Preprocessing for Image Classification**

Preprocess the new dataset to ensure that all images are of the same size and
format. Normalize the pixel values of the images to be between 0 and 1.

**Step 7: Train VGG16**

Train the pretrained VGG16 model on the preprocessed dataset to learn the different
features and patterns of the images. Fine-tune the model parameters to improve the
accuracy of the classification.

**Step 8: Image Classification**

Use the most effective trained VGG16 model to classify the test images into their
respective classes. Evaluate the accuracy of the classification results.










