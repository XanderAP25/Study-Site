---
layout: default
title: Introduction to AI and LLMs
---

## Section One: Introduction to AI and LLMs

Before we explore artificial intelligence, let's briefly consider what intelligence itself is. Intelligence is granted to us by the feat of evolution that is our brain. So, what can we do with our brains? We can analyze information, learn from experience, reason through problems, and make decisions. Artificial intelligence is the ability of a machine to perform tasks that require human intelligence. There are many models that can fuel artificial intelligence, and we will get into the algorithms and structures behind these models later in the article.

A simple example of a model is linear regression analysis, which works by feeding data into the model as training data—predictors and a target outcome. Based on the training data, the model learns the relationships between the predictors and target variables, allowing it to make predictions.

A more advanced example of a machine learning system is the neural network, which draws inspiration from the neurons in our brains. Like biological neurons, artificial neurons receive input, process it, and pass it along “synapses” to other neurons until an output is eventually produced ([^1]). This "input to neuron to output" structure is repeated over and over, sometimes hundreds of thousands of times, just as signals travel through the human brain to form thoughts or trigger physical responses.

<img src="assets/images/Neuron Visual (2).png" style="width:650px;display:block; margin:auto;">
<span style="font-family:Arial; font-size:xx-small;">Illustration of the traversal of input data through neurons.</span>

As shown above, artificial neural networks mimic this process using layers of artificial neurons, or nodes, that process incoming data using mathematical functions. One common function is the sigmoid function ([^2]), which normalizes the input value into a range between 0 and 1, allowing the model to determine whether or not to "activate" a connection. Each node in a layer processes its input and passes its result to the next layer, continuing until a final output is produced.

The bulk of this processing happens in the layers between input and output, or the hidden layers, where data is transformed multiple times before a final decision is made ([^1]). Whether a node passes its output to another is determined by weights—numerical values that represent the strength of the connection between two nodes. These weights are learned during training. For example, a neural network could be trained to identify whether an image depicts an apple or an orange. During training, the network is fed tons of labeled images of apples and oranges. As it processes the images, the weights between nodes are adjusted to recognize the features unique to the fruits. Once trained, this network can classify a new input image as either an apple or an orange based on the features it observed from the training data.

There are many types of neural networks, each developed to tackle increasingly complex and demanding AI tasks. One notable predecessor to today’s most advanced models is the recurrent neural network (RNN), which was once the standard for handling sequential data ([^3]). While we won’t cover RNNs in detail here, it’s useful to understand their limitations, especially because they set the stage for the development of a more powerful alternative: the Transformer model. This architecture is the foundation of many modern AI systems, including OpenAI’s ChatGPT.

Before we can get into the Transformer architecture, it's worth understanding why it replaced RNNs. In a standard neural network, information flows in one direction, from input to output, with no memory of previous inputs. This becomes a major obstacle when dealing with sequential data like text, where context is key. RNNs addressed this by introducing loops that allowed the network to maintain a form of memory, but they struggled with long-term dependencies. For example, when processing the sentence:

**"I love Rad Cat for his academic due diligence and intellect."**

The RNN takes each word as input, one at a time, and treats it as largely independent from the words that came before. When it starts with "I," it has no memory of what’s coming next. If the task were to predict the next word, the network would have to make a guess based solely on "I," without any understanding of the sentence’s broader context. This is a difficult task even for a human, let alone a machine. To handle this, RNNs introduce a loop in their architecture that allows each output to be fed back in as an input, enabling the model to retain information about earlier words as it moves through the sequence. This gives the network a rudimentary memory, so by the time it reaches "intellect," it can recall the presence of "academic" earlier in the sentence and use that relationship to inform its output.

The issue with this method, however, lies in its inefficiency and limited memory. RNNs must process sequences step by step, and as the sequence grows longer, their ability to retain earlier information diminishes. For short sentences like the one above, the memory may hold up well enough, but for longer, more complex sequences, earlier words are often “forgotten” as the network progresses ([^3]). This makes it hard for RNNs to recognize and leverage long-range relationships in text, which are often crucial for understanding nuance, tone, or reference.

The solution to this problem was an entirely new architecture for processing information: the Transformer architecture. The most important concept to understand for the Transformer model is the idea of attention, which is covered at length in *Attention is All You Need* ([^3]).

<img src="assets/images/transformer diagram.png" style="width:300px; height:180px;">
<span style="font-family:Arial; font-size:xx-small;">Illustration of the Transformer architecture ([^3]).</span>

Self-attention is a mechanism that allows a model to look at every word in a sequence at the same time and decide how much attention it should pay to each word in relation to the others ([^3]). This is incredibly helpful when reading sentences because the model can find relationships between words, even if they are far apart. For example, let's look at "I love Rad Cat for his academic due diligence and intellect" again. A self-attention mechanism will process the entire sequence and assign a weight to every pair of words. The mechanism can see that "Rad" has a stronger connection to "Cat" over "love," so the weight will be greater and prioritized when making predictions.

At first glance, this might sound similar to how RNNs process sequences. The key difference is the ability for the self-attention mechanism to consider all words of a sequence at once, as opposed to sequentially reading through it. This removes the performance bottleneck of processing words one by one and also addresses the memory limitations of RNNs that cause them to forget relationships between words separated over a long distance.

Now, there is a limitation with self-attention mechanisms. Instead of just running one self-attention mechanism on the model, multi-head attention runs multiple self-attention mechanisms in parallel with each other. Each mechanism, or head, looks at the sequence from a different perspective and finds different relationships between the words of the sequence. A single self-attention mechanism might only find the relationship between "Rad" and "Cat," while another could find the relationship between "love" and "intellect."

This is enabled by the encoding of the words in the input sequence into a numerical representation, called an embedding, which captures the relationships it has with other words in the sequence. This encoding is split up into a number of smaller sections for each head to focus on. At the end of multi-head attention, these relationships are all added back together, leading to a more complete understanding of the input sequence and a much better ability for predicting the next word in a sequence ([^3]). At a high level, that is how the Transformer model works. It makes use of multi-head attention to look at a sequence from different perspectives to get a more complete understanding of the relationships between the words than it would have had with just a single attention mechanism or through an RNN. For anyone curious about what the actual implementation of this architecture looks like, there are videos out there on that topic ([^4]).

The Transformer model powers modern AI and enables many of the incredible applications we see today. AI is used to translate languages, analyze vast amounts of information to generate summaries, recognize and classify images, create realistic images and videos, and much more. One of the most well-known applications of AI is text generation, where models process a prompt and generate a relevant response. These models, known as Large Language Models (LLMs), specialize in understanding and producing human-like text. Some of the applications mentioned earlier, like text summarization and language translation, also fall under this label. For the rest of this article, we will focus on LLMs, exploring how they work, their benefits, and how you can use them effectively.

<p style="text-align:right;">
    <a href="{{ '/how_llms_can_benefit_you_and_their_challenges.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">How LLMs Can Benefit You and Their Challenges →</a>
</p>

[Footnotes go below here]: #

[^1]: Han, Su-Hyun, et al. “Artificial Neural Network: Understanding the Basic Concepts without Mathematics.” *Https://Doi.Org/10.12779/Dnd.2018.17.3.83*, 17 Sept. 2018, doi.org/10.12779/dnd.2018.17.3.83. 

[^2]: $\sigma(x) = \frac{1}{1+e^{-x}}$ where $x$ is the sum of all the inputs going into the node.

[^3]: Vaswani, Ashish, et al. “Attention Is All You Need.” arXiv.Org, 2 Aug. 2023, arxiv.org/abs/1706.03762. 

[^4]: A video walkthrough of the implementation of the Transformer model: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)

[^5]: Dell'Acqua, Fabrizio, et al. “Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality.” SSRN, 18 Sept. 2023, papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321. 

[^6]: Javaid, Muhammad, et al. "Unlocking the Opportunities through CHATGPT Tool towards Ameliorating the Education System." BenchCouncil Transactions on Benchmarks, Standards and Evaluations, no. 2, 2023, p. 100115. https://doi.org/10.1016/j.tbench.2023.100115.

[^7]: Alqahtani, Tariq, et al. "The Emergent Role of Artificial Intelligence, Natural Learning Processing, and Large Language Models in Higher Education and Research." Research in Social and Administrative Pharmacy, 2023. https://doi.org/10.1016/j.sapharm.2023.05.016.

[^8]: Yao, Renee. “Quicker Cures: How Insilico Medicine Uses Generative AI to Accelerate Drug Discovery.” NVIDIA Blog, 16 Oct. 2024, blogs.nvidia.com/blog/insilico-medicine-uses-generative-ai-to-accelerate-drug-discovery/. 

[^9]: Michael. “The Times Sues OpenAI and Microsoft over A.I. Use of Copyrighted Work.” The New York Times, The New York Times, 27 Dec. 2023, www.nytimes.com/2023/12/27/business/media/new-york-times-open-ai-microsoft-lawsuit.html. 

[^10]: Bender, Emily M., et al. “On the Dangers of Stochastic Parrots: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.” ACM Conferences, 1 Mar. 2021, *dl.acm.org/doi/10.1145/3442188.3445922.*

[^11]: Dong, Qingxiu, et al. “A Survey on In-Context Learning.” arXiv.Org, 5 Oct. 2024, arxiv.org/abs/2301.00234.

[^12]: Banjara, Babina. “Fine-Tuning Large Language Models: A Comprehensive Guide.” Analytics Vidhya, 5 Feb. 2025, www.analyticsvidhya.com/blog/2023/08/fine-tuning-large-language-models/?utm_source=chatgpt.com. 

[^13]: February 12 ACM Talk: "Unlock Hugging Face: Simplify AI with Transformers, LLMs, RAG, Fine-Tuning" with Wei-Meng Lee

[^14]: https://openai.com/policies/row-privacy-policy/

[^15]: <a href = "https://colab.research.google.com/drive/1goVTnNt6FauofB_BAQ2Db4h8uWFvfY1d" target="_blank">Story Generation Using an LLM in a Notebook Environment</a>

[^16]: <a href="https://colab.research.google.com/drive/1ue50VMGv12nzZ6uQNxL6wITtvgJ0nX5V?usp=sharing" target="_blank">Finetuning an LLM on German Fairy Tales</a>

[^17]: <a href = "https://colab.research.google.com/drive/1_q8NFdDmb1_QonBbXd9wk84RiPA8ei8l?usp=sharing#scrollTo=BMzdSSBlq8dq" target="_blank">Text Analysis of Output and Grimms</a>

[^18]: LLMs use tons of VRAM, and require powerful GPUs to be used to their fullest extent. Whether that be output generation, or fine-tuning, you want to have a lot of VRAM at your disposal. You can get away with using CPUs or Apple Silicon in conjunction with your system RAM for running and fine-tuning smaller LLMs, but you will see noticeably less performance and efficiency. Google Colab is a relatively cheap resource to rent powerful GPUs that you can easily use for your AI ambitions. You can even use a free T4 GPU, which offers more power than what the layman would likely have on hand. A good rule of thumb is that you want at least 16 GB of VRAM to do anything significant with moderate sized LLMs (1-3B parameters), and you may want 80GB+ of VRAM if you're fine-tuning larger models.

[^19]: Quantized models have their weights reduced in size from (typically) 32-bit floats to 8-bit integers. This significantly reduces their size and computational overhead while making them less precise in their outputs, meaning that they will have somewhat lower quality generated text. These models still benefit from large training sets and expert optimization, however. For further reading, see Hugging Face’s documentation on quantization: https://huggingface.co/docs/optimum/en/concept_guides/quantization

[^20]: Any web-based frontend for Ollama will require [Docker](https://docs.docker.com/get-started/)  to be installed onto your machine. I recommend having some knowledge of Docker if you're at all interested in software development, so this could be a good introduction to that tool. This front-end is a popular choice for users of Ollama: https://github.com/open-webui/open-webui