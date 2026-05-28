Problem Statement: 

The project focuses on developing an AI-powered adaptive learning platform for differently-abled children to improve educational accessibility and personalized learning experiences. Many children with disabilities such as Dyslexia, Autism Spectrum Disorder (ASD), ADHD, and hearing impairment face difficulties in understanding traditional learning methods due to the lack of personalized and accessible educational systems. The proposed system aims to overcome these challenges by providing multimodal learning support using text, audio, visual content, and sign language assistance. The dataset used for the project contains columns such as Student_ID, Age, Disability_Type, Learning_Level, Topic, Text_Content, Audio_Content, Visual_Content, Sign_Language_Output, Language, Engagement_Score, Difficulty_Level, Assessment_Score, Feedback, Emotion_State, and Learning_Pace.

The Accuracy score for my own problem statemtent Dataset(ML models):

1: Linear Regression = 0.890916917957032
2: Multilinear Regression = 0.865421
3: Logistic Regression = 0.6583333333333333
4: Multinomial Regression = 0.9833333333333333
5: Decisiontree Classifier = 0.9166666666666666
6: SVM = 0.925
7: Randomforest Classifier = 0.875
8: Kmeans(inertia) = 114770.44133975237
9: Naive Bayes = 0.5555555555555556
10:KNN = 0.5833333333333334

I Perfromed the random forest model which give accuracy of 89% , and they perform on the selected features.The image shows the ui of the of model save and perfroming and done using the flask api for the backend.
<img width="1888" height="843" alt="Screenshot 2026-05-22 151132" src="https://github.com/user-attachments/assets/29fe32f4-a002-4a83-b582-c8e66c323990" />

<img width="1897" height="905" alt="image" src="https://github.com/user-attachments/assets/0cd9b878-6afc-4f95-87db-28948eea44a9" />
