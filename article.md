---
layout: default
title: Article
---
# Demystifying AI: A Guide to LLMs

We are now at an awkward point where we have this article section, and I want the article to have its own section and place on the navbar, but I don't have anyway to introduce it differently because that has been done already on the home page.

| Table of Contents |  
| ----------- |  
| [Introduction to AI and LLMs](#introduction-to-ai-and-llms) |  
| [How LLMs Benefit You and Their Challenges](#how-llms-benefit-and-their-challenges) |  
| [Training and Fine-Tuning an LLM](#training-and-fine-tuning-an-llm) |  
| [Making LLMs Work for You](#making-llms-work-for-you) |  

## Introduction to AI and LLMs

Before we explore artificial intelligence, let's briefly consider what intelligence itself is. Intelligence is granted to us by the feat of evolution that is our brain. So, what can we do with our brains? We can analyze information, learn from experience, reason through problems, and make decisions. Artificial intelligence is the ability of a machine to perform tasks that require human intelligence. There are many models that can fuel artificial intelligence, and we will get into the algorithms and structures behind these models later into the article.

A simple example of a model is linear regression analysis, which works by feeding data into the model as training data, predictors and a target outcome. Based on the training data, the model learns the relationships between the predictors and target variables, allowing it to make predictions.

A more advanced example of a machine learning system is the neural network, which draws inspiration from the neurons in our brains. Like biological neurons, artificial neurons receive input, process it, and pass it along “synapses” to other neurons until an output is eventually produced ([^1]). This "input to neuron to output" structure is repeated over and over, sometimes hundreds of thousands of times, just as signals travel through the human brain to form thoughts or trigger physical responses.

<img src="assets/images/Neuron Visual (2).png" style="width:650px;display:block; margin:auto;">
<span style="font-family:Arial; font-size:xx-small; ">Illustration of the traversal of input data through neurons.</span>

As shown above, artificial neural networks mimic this process using layers of artificial neurons, or nodes, that process incoming data using mathematical functions. One common function is the sigmoid function ([^2]), which normalizes the input value into a range between 0 and 1, allowing the model to determine whether or not to "activate" a connection. Each node in a layer processes its input and passes its result to the next layer, continuing until a final output is produced.

The bulk of this processing happens in the layers between input and output, or the hidden layers, where data is transformed multiple times before a final decision is made ([^1]). Whether a node passes its output to another is determined by weights, numerical values that represent the strength of the connection between two nodes. These weights are learned during training. For example, a neural network could be trained to identify whether an image depicts an apple or an orange. During training, the network is fed tons of labeled images of apples and oranges. As it processes the images, the weights between nodes would be adjusted to recognize the features unique to the fruits. Once trained, this network can classify a new input image as either an apple or orange based on the features it observed from the training data.

There are many types of neural networks, each developed to tackle increasingly complex and demanding AI tasks. One notable predecessor to today’s most advanced models is the recurrent neural network (RNN), which was once the standard for handling sequential data ([^3]). While we won’t cover RNNs in detail here, it’s useful to understand their limitations, especially because they set the stage for the development of a more powerful alternative: the Transformer model. This architecture is the foundation of many modern AI systems, including OpenAI’s ChatGPT.

Before we can get into the Transformer architecture, it's worth understanding why it replaced RNN. In a standard neural network, information flows in one direction, from input to output, with no memory of previous inputs. This becomes a major obstacle when dealing with sequential data like text, where context is key. RNNs addressed this by introducing loops that allowed the network to maintain a form of memory, but they struggled with long-term dependencies. For example, when processing the sentence:

**"I love Rad Cat for his academic due dilligence and intellect."**

The RNN takes each word as input, one at a time, and treats it as largely independent from the words that came before. When it starts with "I," it has no memory of what’s coming next. If the task were to predict the next word, the network would have to make a guess based solely on "I," without any understanding of the sentence’s broader context. This is a difficult task even for a human, let alone a machine. To handle this, RNNs introduce a loop in their architecture that allows each output to be fed back in as an input, enabling the model to retain information about earlier words as it moves through the sequence. This gives the network a rudimentary memory, so by the time it reaches "intellect," it can recall the presence of "academic" earlier in the sentence and use that relationship to inform its output.

The issue with this method, however, lies in its inefficiency and limited memory. RNNs must process sequences step by step, and as the sequence grows longer, their ability to retain earlier information diminishes. For short sentences like the one above, the memory may hold up well enough, but for longer, more complex sequences, earlier words are often “forgotten” as the network progresses ([^3]). This makes it hard for RNNs to recognize and leverage long-range relationships in text, which are often crucial for understanding nuance, tone, or reference.

The solution to this problem was an entirely new architecture for processing information, the Transformer architecture. The most important concept to understand for the Transformer model is the idea of attention, which is covered at length in *Attention is All You Need* ([^3]).

<img src="assets/images/transformer diagram.png" style="width:300px; height;180px;">

<span style="font-family:Arial; font-size:xx-small; ">Illustration of the Transformer architecture ([^3]).</span>

Self-attention is a mechanism that allows a model to look at every word in a sequence at the same time and decide how much attention it should pay to each word in relation to the others ([^3]). This is incredibly helpful when reading sentences because the model can find relationships between words, even if they are far apart. For example, let's look at "I love Rad Cat for his academic due dilligence and intellect" again. A self-attention mechanism will process the entire sequence and assign a weight to every pair of words. The mechanism can see that "rad" has a stronger connection to "cat" over "love," so the weight will be greater and prioritized when making predictions.

At first glance, this might sound similar to how RNNs process sequences. The key difference is the ability for the self-attention mechanism to consider all words of a sequence at once as opposed to sequentially reading through it. This removes the performance bottleneck of processing words one by one, and also addresses the memory limitations of RNNs that cause them to forget relationships between words separated over a long distance. 

Now, there is a limitation with self-attention mechanisms. Instead of just running one self-attention mechanism on the model, multi-head attention runs multiple self-attention mechanisms in parallel with each other. Each mechanism, or head, looks at the sequence from a different perspective, and finds different relationships between the words of the sequence. A single self-attention mechanism might only find the relationship between "rad" and "cat" while another could find the relationship between "love" and "intellect." 

This is enabled by the encoding of the words in the input sequence into a numerical representation, called an embedding, which captures the relationships it has with other words in the sequence. This encoding is split up into a number of smaller sections for each head to focus on. At the end of multi-head attention, these relationships are all added back together, leading to a more complete understanding of the input sequence, and a much better ability for predicting the next word in a sequence ([^3]). At a high-level, that is how Transformer model works. It makes use of multi-head attention to look at a sequence from different perspectives to get a more complete understanding of the relationships between the words than it would have had with just a single attention mechanism, or through RNN. For anyone curious on what the actual implementation of this architecture would look like, there are videos out there on that topic ([^4]).

The Transformer model powers modern AI and enables many of the incredible applications we see today. AI is used to translate languages, analyze vast amounts of information to generate summaries, recognize and classify images, create realistic images and videos, and much more. One of the most well-known applications of AI is text generation, where models process a prompt and generate a relevant response. These models, known as Large Language Models (LLMs), specialize in understanding and producing human-like text.  Some of the applications mentioned earlier, like text summarization and language translation also fall under this label. For the rest of this article, we will focus on LLMs, exploring how they work, their benefits, and how you can use them effectively.

## How LLMs Benefit You and Their Challenges

LLMs offer much in the way of enhancing productivity and creativity. At their core, they process text and generate a relevant response. This is a simplified explanation of how LLMs function, but it should be clear the potential that such a technology holds. Imagine having an assistant that can instantly summarize an article or a peer to help brainstorm solutions to complex technical problems. What LLMs have to offer is nothing short of revolutionary.

For both everyday users and professionals, there is some way in which LLMs can make a tangible impact. Imagine a chatbot that can answer questions you might have on a topic of interest or study, but also help troubleshoot any tech issues you might be facing, or a proofreader that can quickly smooth out your writing. If you are considering a project of any kind, an LLM can work as a sounding board that can help identify angles of approach or ideas that you haven't yet considered. In the professional space, management consulatants at Boston Consulting Group have been found to complete tasks 25.1% quicker with 40% better quality in their output just by adopting LLMs into their workflow [^5]. Now, imagine applying that to your own workflow—writing reports, coding, or even managing emails. With AI, you could finish hours of work in minutes, freeing up time for higher-value tasks.

In education, LLMs are transforming how we teach and learn. Think about the time-consuming tasks teachers face daily—creating assessments, developing lesson plans, providing individualized feedback. LLMs can handle all of these efficiently. These tools can develop personalized learning materials like summaries, flashcards, and quizzes tailored to individual students [^6]. What's particularly impressive is how LLMs can make grading more consistent and fair by reducing the subjective biases that naturally creep into human assessment [^7]. Beyond just saving time, these models offer powerful language support—helping students learn new languages, assisting non-native speakers with translations, and even teaching coding skills. What could be most valuable is how using LLMs in classrooms now prepares students for the future state of the workforce. As students learn to craft effective prompts through trial and error, they're building critical thinking skills and technological literacy that will serve them regardless of which careers emerge in our AI-integrated future.

The benefits of AI aren't just limited to any one field or person, they're already reshaping our world in unexpected ways. In the healthcare field specficially, there have been major strides in drug discovery and molecular synthesis with AI models as the drivng forces. Insilico is a company that has made use of generative AI not an LLM exactly but under a similar umbrella to develop a drug that is currently in Phase 2 clinical trials [^7]. Now, they've built an LLM-powered system called Nach0, which designs molecules, predicts their properties, and even suggests ways to synthesize them, potentially cutting years off drug development timelines [^8].

While there is a lot of good that LLMs can do for you and the world at large, they are not a miracle technology with zero drawbacks. One of the most controversal downsides to their very existence is how their training data is obtained and used. We'll be covering this soon, but LLMs need vast amounts of data to get as competent as they've become today. An easy way to get this data is to scrape the entire internet. The owners and creators of digital content are not asked for their consent when their material is used to train AI systems, and this has led to tons of debate on the ethical and legal ramifications of training AI. Currently, there is no law against scraping data from the web and using it to train your models, but some organizations like the *New York Times*, have moved forward with lawsuits against both OpenAI and Microsoft over the use of their copyrighted content to train their models [^9]. This raises an ethical dilemma: should AI companies profit from models trained on content they don't own? Opponents, like *The New York Times*, argue that it disregards creators' rights and exploits their work without compensation. Proponents counter that broad data scraping is essential for AI advancement and the benefits it brings, essentially saying that the ends justify the means.

You might have also wondered how the broad sections of the internet are prepared for AI training. It's not as simple as dumping raw web pages into a model, data needs to be cleaned, filtered, and categorized. Since algorithms aren't perfect at identifying harmful or low-quality content, much of this work falls on human moderators. These workers, often from lower-income countries, are tasked with reviewing massive amounts of disturbing material—graphic violence, hate speech, child exploitation, all to keep AI models "safe" for users. The psychological toll of this work is severe, with many moderators reporting trauma and PTSD-like symptoms from prolonged exposure [^10]. Yet, these workers remain largely invisible, underpaid, and unprotected, raising serious ethical concerns about the unseen labor behind AI advancements [^10].

Beyond data processing, LLMs also struggle with transparency—most training datasets are poorly documented, making it difficult to track biases, misinformation, or potential harm [^10]. This lack of accountability disproportionately affects marginalized groups, as AI models trained on biased data tend to reinforce existing inequalities. From racial and gender biases in generated text to the erasure of underrepresented dialects and cultures, LLMs often reflect and amplify the flaws of the data they consume [^10]. Add to that the environmental cost—training and running large AI models require immense computational power, consuming as much energy as entire cities [^10]. As AI development accelerates, so does its carbon footprint, raising concerns about sustainability and the long-term impact on our planet [^10].

Access to LLMs is not universal—many people, especially in lower-income regions, are excluded from their benefits due to infrastructure limitations, language barriers, or financial costs [^10]. Yet, these same communities often bear the brunt of AI's environmental impact, as the energy-intensive data centers powering these models contribute to climate change and resource consumption that disproportionately affect marginalized populations [^10]. This imbalance raises questions about who truly benefits from AI advancements and who is left dealing with the consequences.

AI is a revolutionary technology, but its development has relied on ethically questionable practices, from exploitative labor to environmental harm. Anyone using it extensively should reflect on whether they're comfortable with these trade-offs. It's an uncomfortable but necessary conversation—one that, by acknowledging AI's downsides, can push us toward a future where these issues are meaningfully addressed.

## Training and Fine-Tuning an LLM

We briefly covered how models are trained in the first section, but let's recap. During training, models process massive sequences of text, analyzing how words relate to one another within a given context. By identifying these relationships, the model learns to predict and generate coherent text based on patterns in its training data. These pre-trained LLMs are what most people will be working with when setting up their own specialized LLM, largely due to the fact that not everyone has the ability to dedicate unfathomable amounts of energy towards the LLM of their dreams. This is where In-Context Learning and fine-tuning come into play.

Both of these techniques influence the LLM's output, but the way each of them work is fundamentally different. Fine-tuning involves retraining the model itself on new data, altering its parameters. In-Context Learning does not require any tweaks to the parameters for the model to become specialized to whatever task you are training it for. Instead, you provide the model with examples, and it learns on the fly as you basically give it an education on whatever topic you are trying to specialize it for [^11]. This is really useful if you want to have a model specialized for tasks such as text classification, sentiment analysis, or language translation, since you can fork over a bunch of examples to the AI, and have it learn how to do these tasks for you. Of course, it isn't as simple as that, but it really isn't that much more complex. Let's take sentiment anaylsis as an example. You want to have your LLM know how to tell whether a review on a movie is positive or negative. To do this, you want to provide a set of demonstration examples for it to follow through a prompt. 

|Prompt|
|------|
|Answer the upcoming prompts using this format: <br>Review: This movie was nothing but a waste of my time and money, I'm done with this franchise. Sentiment: Negative<br>Review: The ultimate conclusion to a story five years in the making. I am speechless. Sentiment: Positive<br>Review: While the visuals and flashy fights were enough to get me through the movie, I just could not engage with the story. Sentiment: Negative|

That is all you need to do for a basic implementation of In-Context Learning, just give the LLM an example to follow, and it will adhere to it to the best of its abilities. Now, there are some complexities to this technique that can let you get the most out of it. Specifically, one must consider the selection of what data is used in the prompt, the format that it is presented in, and the order that it is given to the AI [^11]. We want to provide relevant examples that are formatted in the most consumable format for the AI, and these examples should build on each other logically, like how we don't learn division before we learn how to add, subtract, and multiply. Taking time to work out how you'll educate your AI before actually carrying it out will lead to a better result. For more information on the techniques involved in maximizing In-Context Learning, I would recommend checking out *A Survey on In-Context Learning* and the numerous sources it points to for improved learning practices [^11].

Fine-tuning is more involved than In-Context Learning, requiring modification of the model's parameters by training it on new data that is specialized for the intended function that you want the LLM to perform. Since the LLM is being retrained, the end result is a model that can more effectively carry out the task given to it. The process of fine-tuning requires a dataset relevant to what you want to specialize your LLM for. If that is a sentiment analysis model, then the input dataset could look like this:

|Text|Sentiments|
|---------------|
|This movie was nothing but a waste of my time and money, I'm done with this franchise. | Negative|
|The ultimate conclusion to a story five years in the making. I am speechless. | Positive | 
|While the visuals and flashy fights were enough to get me through the movie, I just could not engage with the story. | Negative |

In a real-world application, you would want a lot more than just three reviews, but this should illustrate the type of data that is fed to models for fine-tuning. 

**(Code sample from either [^12] or [^13] for fine tuning code)**

From here, the model's parameters are tweaked on the dataset, meaning that its understanding of movie review sentiment is being ingrained directly into the model. The key benefit of fine-tuning over In-Context Learning is that fine-tuning creates a more permanent integration of the new infromation into the LLM's structure. Just like a person, LLMs "forget" what they learned after a session ends. Instead of having to feed an LLM the same prompt every time you want to have it learn how to do a specific task, you can finetune a model to permanently remember how to do said task.

**(For specific code examples I am thinking of adapting the fine tuning example from the ACM talk. I was considering talking about RAG here, but have concerns about length. It would be easy to include if we want to go that path though.)**

**(STILL NEED TO GIVE OVERVIEW OF HARDWARE RECS AND RESOURCES TO GET FINETUNING UP AND RUNNING)**

## Making LLMs Work for You

When it comes to making LLMs work for you, there are two methods of doing so. The first that we will cover is accessing LLMs via APIs.

An LLM API lets you connect your computer directly to ultra powerful AI models, like GPT-4 and Llama 2, without actually having to download and run the model on your computer. What this means for you is that you can have the most mediocre hardware on the planet, but as long as you have a wifi connection and the ability to pay the fee to access the model, you have an LLM that you can use and modify to your heart's content. The fee to use these APIs is often not a monthly subscription, but instead a fee per million tokens processed as input and output. For reference, one token is about 4 characters, or about three-fourths of a word. This fee depends on what model you're paying to access, but can be anywhere from \\$30 per million tokens processed as output for OpenAI's GPT-4 Turbo model to \\$2.00 per million tokens processed as output for their smaller GPT-3.5 Turbo model. Prices vary depending on what company's model you use, but there will typically be a price associated with using this larger models.

The cost to use these models presents some concerns when you're just starting out with using LLMs. For example, what constitutes a heavy workload for these models? You could be getting acquainted with your shiny new LLM and suddenly find yourself forking over way more cash than you initially intended because you didn't realize that the multi-hour long conversation or learning session you had with your model required millions of input and output tokens to be processed. There are also privacy concerns associated with using LLM APIs, since they typically store your prompts and outputs to improve their own technology. This data is also used to enforce the company's TOS, prevent illegal activity using their technology, and in specific circumstances can be shared with other parties [^14]. Those circumstances include: at the request of government authorities, to certain service providers and vendors when business needs must be met, and during the diligence process with those assisting in a business transfer[^14]. Since user data is retained, that also opens you up to the possibility of your data being included in a data leak if the company that provides your LLM gets hacked. 

Despite these concerns, LLM APIs are typically at the cutting edge of AI technology. So, if you're looking for the most powerful model for a specific application, using an API might be the right choice.

As mentioned earlier, another option is downloading and running an LLM directly on your own computer. Running LLMs locally gives you more control over your data, as it is stored entirely on your device and never shared with the model's creator. There are also no fees for running a local LLM. While this might sound like the ideal option, there are several caveats to consider. For one, these models require fairly powerful hardware to run efficiently. Although there is no fee to use a local LLM, you will likely need to invest in a powerful PC to get the most out of the technology. Smaller models can be run on more modest hardware, such as a relatively modern laptop, but you'll likely experience slower output and lower quality compared to running top-tier models on a high-performance PC with a strong NVIDIA GPU. (Touch on electricity use and the likelihood of it raising your electricity bill).

**(I should also touch on GPT4ALL and LM Studio here, hugging face API free tier)**

To get an LLM up and running on your own machine, you will need a platform to run them on. One of the most popular platforms for this purpose is [Ollama](https://ollama.com), an open-source tool that allows you to run AI models that you have downloaded through a terminal. Setup is fairly straightforward, just download the application from the website, and follow the steps it gives you to get up and running. The Ollama website also hosts models for you to get up and running, but you can also run models from Hugging Face if there is something there that is more suited to your needs (i just realized that I have not talked about hugging face at all in this entire article). These models are measured by how many parameters they have, and they are formatted like "8B" meaning 8-billion parameters in this case. For weaker computers, like laptops, you will want to use models that are at most 8B. Depending on how powerful your laptop is, you could go higher than that, especially if it has a dedicated NVIDIA GPU. For PCs, you can go higher than this, but will still find a limit around the 30B parameter mark. Despite that being the limit, there are models available for download with upwards of 70B parameters, going into the 600-billions.

To run these models, you would either need to drop a small fortune on a rig dedicated to running massive AI models, or purchase access to a cloud computing service, and use their hardware instead. In a way, this is the middleground between running an LLM on your own computer and using an API to access a company's AI model. You have some of the benefits of running an LLM on your machine, like keeping your data private and the ability to pick and choose specific models you want to use on the fly. And some benefits from LLM APIs, like gaining access to an extremely powerful LLM while not needing a powerful computer. You do have to pay to use these cloud services, but these rates are hourly, instead of usage-based, which could be good or bad depending on what you're doing with your LLM.

**(There is a way to properly end this section, but as a first draft, I think this is a good spot to build from after critique)**

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