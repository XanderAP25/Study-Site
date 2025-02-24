---
layout: default
title: Article
---
# Demystifiying AI: A Guide to LLMs

Depending on who you are, and your interests, different things likely come to mind when you hear "AI". Maybe you think of Skynet from *Terminator*, J.A.R.V.I.S from *Iron Man*, or Cortana from *Halo*. Or perhaps your mind jumps to AI assistants like Siri, Alexa, or Bixby, or the recommendation algorithms on Netflix and YouTube (less likely, but still possible). More generally, AI models like ChatGPT, Gemini, and Grok are probably going to be at the front of your mind. Like magic, these powerful tools were seemingly conjured from nothing and showed themselves to be powerful tools. They can read papers for you, write stories and programs, and are now showing themselves to be capable of logically thinking through problems. Impressive? Absolutely. But not magic.

With some time and effort, you can not only come to understand how AI works, but how to make it work for you. It is with that purpose that this guide exists, to distill the dense and scattered information that is out there on AI into a more concise and easily digestible form.

STILL CONSIDERING HOW I'M GOING TO GIVE A "here's what we're doing and why we're doing it this way" PIECE UP HERE


| Table of Contents |  
| ----------- |  
| [Introduction to AI and LLMs](#introduction-to-ai-and-llms) |  
| [How LLMs Benefit You and Their Challenges](#how-llms-benefit-and-their-challenges) |  
| [Training and Fine-Tuning an LLM](#training-and-fine-tuning-an-llm) |  
| [Running LLMs on Your Own Computer](#running-llms-on-your-own-computer) |  


### Prerequisite

This article is intended for those unfamiliar with the inner workings of artificial intelligence, and the applications of AI that go beyond simply using the web-based solutions afforded by OpenAI and the various other AI companies. No deep math or programming knowledge is required, but a basic familiarity with coding concepts will help. Even if you're new to AI, this guide will give you a solid foundation.

## Introduction to AI and LLMs

Before we explore artifical intelligence, let's briefly consider what intelligence itself is. Intelligence is granted to us by the supercomputer that is our brain. So, what can we do with our brains? We can analyze information, learn from experience, reason through problems, and make decisions.

Artificial intelligence is the ability of a machine to perform tasks that require human intelligence. There are many algorithms and models that can fuel artificial intelligence. What all these algorithms have in common is that they are driven by training on a set of data to perform a certain task. 

A simple example of this is linear regression analysis, which works by feeding data into the algorithm as predictors and a target outcome. Based on the training data, the algorithm learns the relationships between the predictors and target variables, allowing it to make predictions.

**big cool image illustrating linear regression**

A more advanced example would be neural networks, which are inspired by the neurons in our brains. Like our neurons, these artificial neurons take data as an input, then send that data along synapses to another set of neurons until the data is eventually transmitted as an output ([^1]). We skipped a lot of steps, so let's take a moment to look at neural networks more closely.

<img src="assets/images/Neuron Visual.png" style="width:400px; height;400px;">

<span style="font-family:Noto Sans; font-size:xx-small; ">Illustration of the traversal of input data through neurons</span>

As shown above, data is input into an initial set of neurons. From there, the data is sent along the synapses to the next set of neurons, and while it isn't shown here, these traversals between neurons can happen hundreds of thousands of times in humans until we get an output that becomes a thought or process of the body ([^1]). 

Artificial neural networks mimic the neuron-synapase-neuron structure of our brains by essentially creating artificial neurons that are capable of processing and passing information forward. These artificial neurons, or nodes, are hubs where a sigmoid function ([^2]) is run on the sum of all inputs, and the result is sent forward either as an output or to another set of neurons to have the same process carried out again with the the new outputs as inputs ([^1]). 

The next node that the output travels to is determined by the weights between them. These weights are developed during the training of the model, where the training data modifies the strength of the connection between nodes to promote a traversal of data that is more inline with our intended function for the network.

For example, a neural network could be trained to identify whether an item is an apple or an orange. During training, the network is fed tons of labeled images of apples and oranges As it processes the images, the weights between nodes would be adjusted to recognize the features unique to the fruits. Once trained, this network can classify a new input image as either an apple or orange based on the features it observed from the training data.


## How LLMs Benefit You and Their Challenges

## Training and Fine-Tuning an LLM

## Running LLMs on Your Own Computer

[Footnotes go below here]: #

[^1]: Han, Su-Hyun, et al. “Artificial Neural Network: Understanding the Basic Concepts without Mathematics.” *Https://Doi.Org/10.12779/Dnd.2018.17.3.83*, 17 Sept. 2018, doi.org/10.12779/dnd.2018.17.3.83. 

[^2]: $\sigma(x) = \frac{1}{1+e^{-x}}$ where $x$ is the sum of all the inputs going into the node.