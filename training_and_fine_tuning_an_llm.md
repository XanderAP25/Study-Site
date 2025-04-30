---
layout: default
title: Training and Fine-Tuning an LLM
---

## Section Three: Training and Fine-Tuning an LLM

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

<p style="text-align:left;">
    <a href="{{ '/how_llms_benefit_you_and_their_challenges.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">← How LLMs Benefit You and Their Challenges</a>
    <span style="float:right;">
        <a href="{{ '/making_llms_work_for_you.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">Making LLMs Work For You →</a>
    </span>
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