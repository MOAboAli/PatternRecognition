# PatternRecognition
Course Pattern Recognition, Using Different algorithm on  Iris Dataset


Dataset: Iris Dataset

Using python programing language.

The iris dataset contains the following data

    •	50 samples of 3 different species of iris (150 samples total)
    •	3 Classes :  "0": setosa,"1": versicolor,"2": virginica
    •	4 Features , dataset is 150 * 4 Matrix
    •	Features: sepal length, sepal width, petal length, petal width
    
    
 
![image](https://user-images.githubusercontent.com/26040529/113583958-9525b000-962a-11eb-9022-382bf086b7bc.png)

Algorithms:


PCA :

Converting 4 dimensional to 2 dimensional

![image](https://user-images.githubusercontent.com/26040529/113584088-c43c2180-962a-11eb-949d-b8d4c82b4850.png)

Converting 2 dimensional to 1 dimensional:

![image](https://user-images.githubusercontent.com/26040529/113584129-d027e380-962a-11eb-8b57-c80eca7aa19b.png)

Red  is class 0 , green is class 1 , blue is class 2



LDA :
Converting 4 dimensional to 2 dimensional

![image](https://user-images.githubusercontent.com/26040529/113584233-efbf0c00-962a-11eb-9071-8882d837f932.png)


Converting 2 dimensional to 1 dimensional:

![image](https://user-images.githubusercontent.com/26040529/113584262-f9487400-962a-11eb-84df-d67c6c6acdfb.png)



Perceptron:

•	First, reduce features to 2 dimensions using LDA:
•	Learning Rate=0.01
•	Iteration =15
•	Weights after training [ 77.90162435 433.32165847  35.24875973]



Multi-Layer Perceptron :



    •	First, reduce features to 2 dimensions using LDA:
    •	Iteration : 1000
    •	Input Layer : 2 nodes
    •	One hidden layer : 100 nodes
    •	Out Layer: 3 nodes
    •	Loss Rate per every 100:
            o	number of epoch 0 loss 1.058013916015625
            o	number of epoch 100 loss 0.7257269024848938
            o	number of epoch 200 loss 0.6850542426109314
            o	number of epoch 300 loss 0.6629276275634766
            o	number of epoch 400 loss 0.6485032439231873
            o	number of epoch 500 loss 0.6379558444023132
            o	number of epoch 600 loss 0.6296232342720032
            o	number of epoch 700 loss 0.6226478219032288
            o	number of epoch 800 loss 0.6165931224822998
            o	number of epoch 900 loss 0.6112104058265686



K-Means
    •	First, reduce features to 2 dimensions using LDA:
    •	K_Classes =3
    •	Output Centroid Center:
            o	[-7.60759993  0.21513302]
            o	 [ 5.75542597  0.52437413]
            o	 [ 1.77251575 -0.76530064]

![image](https://user-images.githubusercontent.com/26040529/113584390-2137d780-962b-11eb-8fa9-0c54a93a4394.png)



Test Cases :

	![image](https://user-images.githubusercontent.com/26040529/113584487-40366980-962b-11eb-92d4-c7944b9c4219.png)



![image](https://user-images.githubusercontent.com/26040529/113584511-44fb1d80-962b-11eb-8fc1-07a94d757a58.png)

				







