---
layout: default
title: Article
---
# Demystifiying AI: A Guide to LLMs

Depending on who you are, and your interests, different things likely come to mind when you hear "AI". Maybe you think of Skynet from *Terminator*, J.A.R.V.I.S from *Iron Man*, or Cortana from *Halo*. Or perhaps your mind jumps to AI assistants like Siri, Alexa, or Bixby, or the recommendation algorithms on Netflix and YouTube (less likely, but still possible). More generally, AI models like ChatGPT, Gemini, and Claude are probably going to be at the front of your mind. Like magic, these powerful tools were seemingly conjured from nothing and showed themselves to be incredibly useful. They can read papers for you, write stories and programs, and are now showing themselves to be capable of logically thinking through problems. Impressive? Absolutely. But not magic.

With some time and effort, you can not only come to understand how AI works, but how to make it work for you. It is with that purpose that this guide exists, to distill the dense and scattered information that is out there on AI into a more concise and easily digestible form.

STILL CONSIDERING HOW I'M GOING TO GIVE A "here's what we're doing and why we're doing it this way" PIECE UP HERE


| Table of Contents |  
| ----------- |  
| [Introduction to AI and LLMs](#introduction-to-ai-and-llms) |  
| [How LLMs Benefit You and Their Challenges](#how-llms-benefit-and-their-challenges) |  
| [Training and Fine-Tuning an LLM](#training-and-fine-tuning-an-llm) |  
| [Running LLMs on Your Own Computer](#running-llms-on-your-own-computer) |  


### Prerequisite

This article is intended for those unfamiliar with the inner workings of artificial intelligence, and the AI applications beyond just using tools like ChatGPT, Gemini, or Claude. No deep math or programming knowledge is required, but a basic familiarity with coding concepts will help. Even if you're new to AI, this guide will give you a solid foundation.

## Introduction to AI and LLMs

Before we explore artifical intelligence, let's briefly consider what intelligence itself is. Intelligence is granted to us by the supercomputer that is our brain. So, what can we do with our brains? We can analyze information, learn from experience, reason through problems, and make decisions. Artificial intelligence is the ability of a machine to perform tasks that require human intelligence. There are many algorithms and models that can fuel artificial intelligence. What all these algorithms have in common is that they are driven by training on a set of data to perform a certain task. 

A simple example of this is linear regression analysis, which works by feeding data into the algorithm as predictors and a target outcome. Based on the training data, the algorithm learns the relationships between the predictors and target variables, allowing it to make predictions.

**big cool image illustrating linear regression**

A more advanced example would be neural networks, which are inspired by the neurons in our brains. Like our neurons, these artificial neurons take data as an input, then send that data along synapses to another set of neurons until the data is eventually transmitted as an output ([^1]). We skipped a lot of steps, so let's take a moment to look at neural networks more closely.

<img src="assets/images/Neuron Visual (2).png" style="width:650px;display:block; margin:auto;">
<span style="font-family:Arial; font-size:xx-small; ">Illustration of the traversal of input data through neurons.</span>

As shown above, data is input into an initial set of neurons. From there, the data is sent along the synapses to the next set of neurons, and while it isn't shown here, these traversals between neurons can happen hundreds of thousands of times in humans until we get an output that becomes a thought or process of the body ([^1]). 

Artificial neural networks mimic the neuron-synapase-neuron structure of our brains by essentially creating artificial neurons that are capable of processing and passing information forward. These artificial neurons, or nodes, are hubs where a sigmoid function ([^2]) is run on the sum of all inputs, and the result is sent forward either as an output or to another set of neurons to have the same process carried out again with the the new outputs as inputs ([^1]). Between the input and output layers of a neural network, there is the hidden layer, which gets its name because we typically do not see what is going on in there. The bulk of the computation is done here as the input data is processed and ran through the sigmoid functions of the nodes until it becomes output.

The next node that the output travels to is determined by the weights between them. These weights are developed during the training of the model, where the training data modifies the strength of the connection between nodes to promote a traversal of data that is more inline with our intended function for the network. For example, a neural network could be trained to identify whether an item is an apple or an orange. During training, the network is fed tons of labeled images of apples and oranges. As it processes the images, the weights between nodes would be adjusted to recognize the features unique to the fruits. Once trained, this network can classify a new input image as either an apple or orange based on the features it observed from the training data.

There are many types of neural networks that have been developed to allow for increasingly complex and performant AI applications. The previous standard for neural networks in AI was the recurrent neural network, and while we are not going to be covering this structures at length, it is important to at least be aware of its existence [^3]. Instead, we will be covering the Transformer model architecture, which has set the groundwork for the wildly impressive AI models, like ChatGPT, that we have today.

<img src="assets/images/transformer diagram.png" style="width:300px; height;180px;">

<span style="font-family:Arial; font-size:xx-small; ">Illustration of the Transformer architecture ([^3]).</span>

Before we can get into the Transformer architecture, let's loop back to recurrent neural networks. As we saw in the neural network diagram, information can only move forward in a neural network, from input to output. This presents a problem when we feed it information that is in a sequence, like text for example. Each word in a sequence is taken as input, so there's no way for the neural network to remember what came before a word. Another way to put it is that there's no context for what lead up to the current word. As an example, let's consider the following sentence:

**"The cat in the hat."**

If we feed this sentence to a neural network, it takes each word as an input, separate from all the others. So, it reads through the sentence and it gets to "the", and it has no idea what came before it. If given a task of figuring out what comes after "the", the neural network would have to make a wild prediction based off of the one word that it is on. That is a task that a human would be unable to do, much less a neural network. For a neural network to do a task such as this, it needs memory, and that is where recurrent neural networks come in. If you are familiar with recursion, then you might see where we're going with this. Recursive neural networks (RNNs) introduce a loop into their architecture that allows them to take an output, and reuse it as an input, retaining information on a sequence as it processes it. This gives the neural network a form of memory to work with when processing sequences, so when it is given a sequence like "The cat in the ___," it will have a much better chance at correctly predicting the next word in the sequence.

The problem with this approach is the overhead that comes with recursively checking the input sequence repeatedly. It's not that much of an issue when you have shorter sequences like the earlier example, but for larger sequences, there is going to be an impact on the efficiency and performance of the RNN from repeatedly going through the sequence from beginning to end as the network processes the data. Just like our own sometimes faulty memories, the memory of RNNs have a limit when sequentially processing data, and will "forget" information from earlier in the sequence ([^3]). This presents a problem when attempting to understand relationships in data that are far apart.

The solution to this problem was an entirely new architecture for processing information, the Transformer architecture. The most important concept to understand for the Transformer model is the idea of attention, it's all you need.  

Self-attention is a mechanism that allows a model to look at every word in a sequence at the same time and decide how much attention it should pay to each word in relation to the others ([^3]). This is incredibly helpful when reading sentences because the model can find relationships between words, even if they are far apart. For example, let's look at "The cat in the hat." A self-attention mechanism will process the entire sequence and assigns a weight to every pair of words. The mechanism can see that "hat" has a stronger connection to "cat" over "in," so the weight will be greater and be prioritized when making predictions.

At first glance, this might sound similar to how RNNs process sequences. The key difference is the ability for the self-attention mechanism to consider all words of a sequence at once as opposed to sequentially reading through it. This removes the performance bottleneck of processing words one by one, and also addresses the memory limitations of RNNs that cause them to forget relationships between words separated over a long distance. 

**(I need some image to break up the text slog in this section)**

Now, there is a limitation with self-attention mechanisms. They can only focus on one type of relationship in a sequence at a time. So, if there is a case where there are multiple relationships between words in a sequence, a self-attention mechanism is likely to miss out on those other relationships. This is where multi-head attention comes in.

Instead of just running one self-attention mechanism on the model, multi-head attention runs multiple self-attention mechanisms in parallel with each other. Each mechanism, or head, looks at the sequence from a different perspective, and finds different relationships between the words of the sequence. A single self-attention mechanism might only find the relationship between "cat" and "hat" while another could find the relationship between "in" and "the." 

This is enabled by the encoding of the words in the input sequence into a numerical representation, called an embeding, which captures the relationships it has with other words in the sequence. This encoding is split up into a number of smaller sections for each head to focus on. At the end of multi-head attention, these relationships are all added back together, leading to a more complete understanding of the input sequence, and a much better ability for predicting the next word in a sequence ([^3]). At a high-level, that is how the Transformer model works. It makes use of multi-head attention to look at a sequence from different perspectives to get a more complete understanding of the relationships between the words than it would have had with just a single attention mechanism, or through RNN. For the purposes of understanding how the Transformer model works, this is enough, but for anyone curious on what the actual implementation of this architecture would look like, there are videos out there on that topic ([^4]).

The Transformer model powers modern AI and enables many of the incredible applications we see today. AI is used to translate languages, analyze vast amounts of information to generate summaries, recognize and classify images, create realistic images and videos, and much more. One of the most well-known applications of AI is text generation, where models process a prompt and generate a relevant response. These models, known as Large Language Models (LLMs), specialize in understanding and producing human-like text.  Some of the applications mentioned earlier, like text summarization and language translation also fall under this label. For the rest of this article, we will focus on LLMs, exploring how they work, their benefits, and how you can use them effectively.

**(There is a better way to end this section and I want to spend some time improving things, but I need to move on)**

## How LLMs Benefit You and Their Challenges

LLMs offer much in the way of enhancing productivity and creativity. At their core, they process text and generate a relevant response. This is a simplified explanation of how LLMs function, but it should be abundantly clear the potential that such a technology holds. For personal use, LLMs offer writing assistance with drafting emails and improving your grammar and style, summarization of large swathes of text, chatbots for getting assistance and troubleshooting, assistance with writing code, and so much more. The amount of applications available to LLMs are so great that we could be here all day listing them out and finding new uses for them.

## Training and Fine-Tuning an LLM

## Running LLMs on Your Own Computer

[Footnotes go below here]: #

[^1]: Han, Su-Hyun, et al. “Artificial Neural Network: Understanding the Basic Concepts without Mathematics.” *Https://Doi.Org/10.12779/Dnd.2018.17.3.83*, 17 Sept. 2018, doi.org/10.12779/dnd.2018.17.3.83. 

[^2]: $\sigma(x) = \frac{1}{1+e^{-x}}$ where $x$ is the sum of all the inputs going into the node.

[^3]: Vaswani, Ashish, et al. “Attention Is All You Need.” arXiv.Org, 2 Aug. 2023, arxiv.org/abs/1706.03762. 

[^4]: A video walkthrough of the implementation of the Transformer model: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)