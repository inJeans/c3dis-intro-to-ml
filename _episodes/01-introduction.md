---
title: "Introduction"
teaching: 60
exercises: 30
questions:
- "What is Machine Learning?"
- "Why would you use it?"
- "When should you use it and when should you avoid it?"
objectives:
- "Understand the distinction between ML and traditional programming"
- "Understand why it can be useful"
- "Know limitations and requirements for implementation"
- "Identify different types of ML"
keypoints:
- "ML algorithms learn from data instead of being human-programmed"
- "A large amount of quality data is essential for ML"
---

## Why to ML?

> ## Fundamental
> Science is fundamentally data driven.
{: .callout}

| ![](../fig/investigator.png) | 
|:--:| 
| RV Investigator |

| ![](../fig/askap.png) | 
|:--:| 
| Australian Square Kilometer Array Pathfinder |

| ![](../fig/energy.png) | 
|:--:| 
| Energy Sector |

> ## Fast
> Machine (Deep) Learning has become the poster child for fast, scalable, exascale compute.
{: .callout}

| ![](../fig/summit.png) | 
|:--:| 
| Summit Supercomputer |

> ## Smart
> Machine Learning is able to uncover insights in very complex systems without knowledge of the fundamental underlying models.
{: .callout}

| ![](../fig/aiml.png) | 
|:--:| 
| Machine Learning offers a purely data driven approach to scientific discovery. |

## What is ML?

FUJI or PINK LADY

|![](../fig/fuji-01.png)|![](../fig/pink_lady-01.png)|![](../fig/fuji-04.png)|![](../fig/pink_lady-03.png)|
|![](../fig/pink_lady-02.png)|![](../fig/fuji-03.png)|![](../fig/fuji-05.png)|![](../fig/pink_lady-04.png)|
|![](../fig/fuji-02.png)|![](../fig/fuji-06.png)|![](../fig/pink_lady-06.png)|![](../fig/pink_lady-05.png)|

- Traditional models are explicitly programmed
-	ML algorithms are learned
-	Therefore data is very important
-   Predicition vs. understanding
-   Interpretability

<!-- Exercises: -->
<!-- -	Give some examples of explicit programming -->
<!-- -	Give some examples of machine learning -->

## When to ML?
-	Data requirements: LOTS!!! The more the better. *Very* generally speaking you want on the order of 10-100 samples for each feature in your data. So for example in the image processing case where we have 128x128x3 pixels we would want on the order of 500,000 -> 5,000,000 samples.
-	Compute requirements: In the case of deep learning the field is mostly focused on GPUs. 
-	Desired outcomes: Supervised machine learning in particular isn't overly useful for providing insight into data. It is mostly designed for inferencing or predictions. Unsupervsied learning on the other hand offers the opportunity to uncover patterns and relationships in data that were previously unseen.

## Types of ML
-	Supervised (labelled data)
     -  Classification
     -  Regression
![](../fig/cat_dog.png)
-	Unsupervised (unlabelled data)
     - Clustering / segmentation
     - Dimension reduction
![](../fig/cyber_security.png)
- Reinforcement Learning
     - Environemt, state, action 
![](../fig/mit_robot.png)
-	Deep learning
![](../fig/deep_learning.png)


{% include links.md %}

