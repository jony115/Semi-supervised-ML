Here’s a refined and professional version of your report summary with improved structure, grammar, and flow:

---

**Objective:**  
This report aims to analyze consumer emotions and brand perceptions using textual reviews. By categorizing text data into distinct emotion classes and evaluating reviews across different countries, the study leverages Natural Language Processing (NLP) techniques and machine learning algorithms to uncover patterns, sentiments, and trends that offer deeper insights into customer brand preferences.

**Methodology & Findings:**  
A Supervised Support Vector Machine (SVM) classifier was evaluated on labeled text data to assess its effectiveness in detecting emotional categories. The model achieved a micro-averaged F1-score of approximately **0.52**, revealing moderate performance. The confusion matrix highlighted challenges in distinguishing between certain emotions, with notable misclassifications. The SVM classifier performed relatively well in identifying *'joy'* and *'neutral'* emotions, showing higher precision and recall. However, it struggled significantly with categories such as *'anger'* and *'surprise'*, which recorded low precision and recall scores. Emotions like *'disgust'*, *'fear'*, and *'sadness'* demonstrated moderate classification performance.

A Random Forest classifier was also implemented, outperforming the SVM model with an overall accuracy of approximately **57.9%**. According to the classification report and confusion matrix:
- *'Joy'* achieved the best performance, with an F1-score of **0.83**, indicating excellent precision and recall.
- Emotions such as *'disgust'* and *'surprise'* showed moderate results, with F1-scores of **0.50** and **0.65**, respectively.
- *'Anger'* and *'sadness'* were more difficult to classify accurately, receiving F1-scores of **0.31** and **0.41**.
- While *'fear'* and *'surprise'* were identified with reasonable success, the classifier struggled with *'neutral'*—despite high recall—suggesting possible overfitting or confusion with other emotional categories.

**Conclusion:**  
The analysis demonstrates that while machine learning models like SVM and Random Forest can effectively identify certain emotions from textual data—especially *'joy'*—they encounter difficulties with subtler or less distinct emotional categories. These findings highlight the complexity of emotion detection in natural language and point toward opportunities for further model optimization and feature enhancement.

---

Let me know if you'd like this adjusted into a PowerPoint summary or an academic-style abstract.
