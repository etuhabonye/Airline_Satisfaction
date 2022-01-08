### Team Members 
Rachel Liu
Emma Tuhabonye
Fushu Beauthier

### Problem Statement

Passengers have given a US airline (unnamed) their customer feedback, via a survey asking them to rate the quality of various services, along with their overall satisfaction ratings (satisfied vs. neutral/dissatisfied).

The problem we are interested in solving is trying to predict future passengers’ overall satisfaction ratings for this airline, based on different combinations of ratings for the various services related to the whole flight experience. In addition, we would like to find out which variables have the greatest effect on overall satisfaction. 

This would be a useful problem to solve as it would allow the airline company to use a model to vary satisfaction ratings of different services and estimate overall satisfaction, allowing the company to predict in advance how many passengers they would gain or lose as a consequence. It would also be useful for the airline to know which variables are most important in keeping overall satisfaction high, so that the airline company knows which services (variables) to focus on keeping high/improving. 

### Data Description 

The data set came from Kaggle (https://www.kaggle.com/johndddddd/customer-satisfaction)
and contains three different kinds of variable types. The source does not provide much information on how the data was collected, but as far as we can tell, the airline obtained passenger feedback through a satisfaction survey. The variables included:

Categorical variables - these variables are not satisfaction ratings but give information about the passengers.
Satisfaction (satisfied or neutral/dissatisfied)
Gender (male or female)
Customer Type (loyal or disloyal)
Type of travel (business travel or personal travel)
Class (eco, eco plus or business)

Continuous numerical variables - these variables are not satisfaction ratings but give information about the passengers and the airline’s flights.
Age (mean = 39.43, max = 85, and min = 7)
Flight distance (mean = 1981.01 mi, max = 6951 mi, and min = 50 mi)
Departure Delay in Minutes (mean = 14.64 mins, max = 1592 mins, min = 0 mins)
Arrival Delay in Minutes (mean = 15.09 mins, max = 1584 mins, and min = 0 mins)

Discrete numerical variables - these variables are satisfaction ratings, all ranging from 0-5, based on passengers’ level of satisfaction with the services provided.
Departure/Arrival Time Convenient (mean = 2.99)
In-flight Wi-fi service: (0=Not Applicable; mean = 3.25)
In-flight entertainment (mean = 3.38)
Ease of Online booking (mean = 3.47)
Online boarding/check-in (mean = 3.35) 
Online support (mean = 3.52)
Food and drink (mean = 2.85)
Seat comfort (mean = 2.84)
On-board service (mean = 3.47)
Leg room service (mean = 3.49)
Cleanliness (mean = 3.71)
Check-in service (mean = 3.34)
Gate location (mean = 2.99)
Baggage handling (mean = 3.70)


### Data Preprocessing 
Describe any variable transformations, treatment of missing values, recoding and any other data manipulations completed prior to applying machine learning techniques. 
Include the code that reads and munges the data (in Rmd file)

### Machine Learning Approaches
For this project, we used 5 different Machine Learning techniques. The first was a Random Forest Classifier. This technique uses individual decision trees that each return a class prediction. The model then chooses the class with the most predictions. For our Random Forest model we decided to do a 15 fold cross validation method, meaning that our data gets resampled into 15 different subgroups of the data where we train and test that model on these different subgroups. We also decided to use the hyperparameter mtry to decide the number of features that will be sampled at each split. 

The next model we used is KNN. It assumes the similarity between the new cases/data and available cases and puts the new case into a category that is the most similar to the available categories. We made k equal to 6 which means a new data point is classified by the majority votes from it’s 6 neighbors. 
 
Classification Trees consist of binary decision trees that lead to a predicted class (In our case that would be satisfied or neutral/dissatisfied. For each tree, the model adds up the misclassification at every terminal node and then multiplies the number of splits and the penalty term (determined through cross validation) and adds to the misclassification. The cp is the scaled version of the penalty rate over the misclassification rate. In short, the cp is the stopping parameter. Our stopping parameter was 0.002437241.

Extreme Gradient Boosting created a more regularized model to control overfitting compared to just Gradient Boosting. Gradient refers to the technique where we minimize the error. The Gradient technique creates predictions based on a learning rate (eta). We update the predictions so the sum of our residuals is close to 0 and the predicted values are close to actual values. For our data, we made the maximum depth of the tree by 3, had our learning rate at .4, and the number of features taken at each tree as a random sample was 60% of the features. 

Artificial Neural Networks try to simulate brain cells inside a computer so it can recognize patterns and make decisions in a human-like way. It has an input layer, hidden layer, and an output layer. In the input layer, the input nodes are information the model is provided to learn and make conclusions from. For our project, this means the features of our data set and what satisfaction class they come out to. The Hidden Layer is where the computations are performed on the input data. We set our model to have 9 hidden layers. Finally, the output layer is the conclusions the model came to. We set our regularization parameter to 0.01 to prevent overfitting.

### Results

The Random Forest model had the highest sensitivity of .94. We chose this as our best model because we wanted to focus on keeping the customer retention rates high. Knowing the true positive rate of the data, we can strategize how to maintain the customers we have already satisfied and not worry as much about the ones we could not impress the first time. The variables that we found the most important in the Random Forest model was the flight entertainment, seat comfort, and ease of online booking. The Receiver Operating Characteristics had an area under the curve of 1. Considering we had over 10,000 data points in the training data, we believe that the model was trained very well. Looking at our gain and lift charts, we see that we would only have to reach out to 50% of our positive reacting customers and we would be able to retain 90% of our satisfied customers. 


### Discussion 

As mentioned above, we were aiming for our models to have high sensitivity in order to correctly predict which passengers would be satisfied with the airline company, as we think the airline company is more likely to make efforts to retain customers who are already satisfied with them than those who are not. 
We found that of our five different models, Random Forest and Extreme Gradient Boost had the most satisfactory sensitivity and accuracy rates (see results). Thus, we think that if the company were to use one of these models, they would be able to predict quite well future passenger satisfaction and consequently how many customers they think they would lose or gain when altering certain variables, which was the aim of our problem!

We were also able to solve the second part of our problem and find out the most important variables in common among all of our models that affect overall passenger satisfaction, namely seat comfort and in-flight entertainment. This would be useful for both future passengers and the airline company. For future passengers, it would be useful as they could immediately consider the flight’s level of seat comfort (through qualitative judgement, maybe looking at the website) and whether they think the in-flight entertainment would be satisfactory in order to decide whether they would be satisfied overall with the flight, thus helping them decide whether to book with that airline or not. For the airline company, these findings are useful as they can now consider focusing on maintaining a high level of seat comfort (for example, reviewing materials, technologies every few months) and keeping up to date with the latest or most popular entertainment products to continuously improve overall satisfaction ratings, or keep them high.

We had some difficulties studying the correlation/collinearity between variables, which is highly probable to affect the results of our models. We would strongly encourage other researchers who want to use this work for further studies to take this into account and build upon it. Lastly, we found that airlines often use these kinds of satisfaction surveys to identify which factors (variables) are relationship drivers (those that impact customer relationships) and which factors are industry drivers (general reliability of services, value for money etc.) for the airline company. Airline companies then plot these on a graph with “Likelihood to return” on the y-axis and the mean satisfaction rating on the x-axis. Plotting these variables on such a graph helps the airline company understand exactly which services they should prioritise, improve on, maintain and leverage. The following figure is helpful in visualising 
how airline companies achieve this. 

Figure 1. variable mean satisfaction ratings are plotted on the x-axis against “Likelihood to return” 
on the y-axis.

Thus, it could be quite useful for airline companies to have both this kind of graph and a model such as ours to cross-check what the most important variables to focus on are. 

### References 

Passenger Satisfaction: US Airline Passenger Satisfaction. (2015). kaggle. https://www.kaggle.com/johndddddd/customer-satisfaction 
How Airlines Use NPS to Improve Their Customer Satisfaction Ratings. (2018, August 22). Retently. https://www.retently.com/blog/airline-satisfaction/
