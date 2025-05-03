---
layout: default
title: Training and Fine-Tuning an LLM
---

## Section Four: Training and Fine-Tuning an LLM

We briefly covered how models are trained in the first section, but let's recap. During training, models process massive sequences of text, analyzing how words relate to one another within a given context. By identifying these relationships, the model learns to predict and generate coherent text based on patterns in its training data. These pre-trained LLMs are what most people will work with when setting up their own specialized LLM, largely because not everyone has the ability to dedicate unfathomable amounts of energy toward training their ideal model. This is where in-context learning (ICL) and fine-tuning come into play.

Both of these techniques influence the LLM's output, but they operate in fundamentally different ways. Fine-tuning involves retraining the model itself on new data, altering its parameters. In-context learning does not require any tweaks to the model's parameters for it to specialize in the task at hand. Instead, you provide the model with examples, and it learns dynamically as you educate it on the desired topic ([^11]). This approach is particularly useful for tasks such as text classification, sentiment analysis, or language translation since you can supply examples for the AI to follow, allowing it to learn how to complete the tasks.  

Let's take sentiment analysis as an example. Suppose you want your LLM to determine whether a movie review is positive or negative. To do this, you provide a set of demonstration examples within a prompt:  

|Prompt|  
|------|  
|Answer the upcoming prompts using this format: <br>Review: This movie was nothing but a waste of my time and money. I'm done with this franchise. Sentiment: Negative<br>Review: The ultimate conclusion to a story five years in the making. I am speechless. Sentiment: Positive<br>Review: While the visuals and flashy fights were enough to get me through the movie, I just could not engage with the story. Sentiment: Negative|  

That is all you need for a basic implementation of ICL—just give the LLM an example to follow, and it will adhere to it as best it can. However, there are nuances to this technique that can maximize its effectiveness. Specifically, one must consider the selection of data used in the prompt, the format in which it is presented, and the order in which it is given to the AI ([^11]). We want to provide relevant examples formatted in an accessible manner and ensure they build upon one another logically. Taking time to plan your approach before executing it will lead to better results. For more details on techniques to optimize in-context learning, I recommend reading *A Survey on In-Context Learning* and the sources it references for improved learning practices ([^11]).  

You can see how I used ICL to enhance an LLM's story generation capabilities in my story generation code notebook ([^15]).  

Fine-tuning is more involved than ICL, requiring modification of the model's parameters by training it on new data specialized for the intended function. Since the LLM undergoes retraining, the result is a model that more effectively executes the desired task. The fine-tuning process requires a relevant dataset. If the goal is sentiment analysis, the dataset might resemble the following:  

|Text|Sentiment|  
|----|---------|  
|This movie was nothing but a waste of my time and money. I'm done with this franchise. | Negative|  
|The ultimate conclusion to a story five years in the making. I am speechless. | Positive |  
|While the visuals and flashy fights were enough to get me through the movie, I just could not engage with the story. | Negative |  

In a real-world application, you would need far more than just three reviews, but this serves as an example of the type of data provided for fine-tuning.  

From here, the model’s parameters are adjusted based on the dataset, ingraining an understanding of movie review sentiment directly into the model. The key advantage of fine-tuning over ICL is that fine-tuning integrates the new information permanently into the LLM's structure. Just like a person, LLMs "forget" what they learned after a session ends. Instead of feeding an LLM the same prompt every time you want it to complete a specific task, you can fine-tune a model to ensure it remembers how to perform the task consistently.  

You can see how I used a dataset consisting of German fairy tales to fine-tune an LLM in my fine-tuning code notebook ([^16]). If you are interested in comparing the fine-tuned model to a baseline model using ICL, I recommend reviewing my text analysis code notebook ([^17]). If not, just know that fine-tuning in this case led to generated stories that more closely resembled traditional German fairy tales than those produced via ICL. However, this increased accuracy came at a significantly higher cost in terms of efficiency. The stories generated using ICL prompts were still relatively close to the fine-tuned model’s outputs in structure and theme, meaning that, depending on your needs, ICL may be preferable to fine-tuning despite some loss in quality. Fine-tuning is a costly process, and while I discuss it further in my case study notebooks, I encourage you to explore the references below for more details ([^18]).

<p style="text-align:left;">
    <a href="{{ '/making_llms_work_for_you.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">← Making LLMs Work For You</a>
    <span style="float:right;">
        <a href="{{ '/conclusion.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">Conclusion →</a>
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

[^15]: <a href = "https://colab.research.google.com/drive/1goVTnNt6FauofB_BAQ2Db4h8uWFvfY1d" target="_blank">Story Generation Using an LLM in a Notebook Environment</a>

[^16]: <a href="https://colab.research.google.com/drive/1ue50VMGv12nzZ6uQNxL6wITtvgJ0nX5V?usp=sharing" target="_blank">Finetuning an LLM on German Fairy Tales</a>

[^17]: <a href = "https://colab.research.google.com/drive/1_q8NFdDmb1_QonBbXd9wk84RiPA8ei8l?usp=sharing#scrollTo=BMzdSSBlq8dq" target="_blank">Text Analysis of Output and Grimms</a>

[^18]: LLMs use tons of VRAM, and require powerful GPUs to be used to their fullest extent. Whether that be output generation, or fine-tuning, you want to have a lot of VRAM at your disposal. You can get away with using CPUs or Apple Silicon in conjunction with your system RAM for running and fine-tuning smaller LLMs, but you will see noticeably less performance and efficiency. I mentioned this in the third section, but Google Colab is a relatively cheap resource to rent powerful GPUs that you can easily use for your AI ambitions. You can even use a free T4 GPU, which offers more power than what the layman would likely have on hand. A good rule of thumb is that you want at least 16 GB of VRAM to do anything significant with moderate sized LLMs (1-3B parameters), and you may want 80GB+ of VRAM if you're fine-tuning larger models.

[^19]: Quantized models have their weights reduced in size from (typically) 32-bit floats to 8-bit integers. This significantly reduces their size and computational overhead while making them less precise in their outputs, meaning that they will have somewhat lower quality generated text. These models still benefit from large training sets and expert optimization, however. For further reading, see Hugging Face’s documentation on quantization: https://huggingface.co/docs/optimum/en/concept_guides/quantization

[^20]: Any web-based frontend for Ollama will require [Docker](https://docs.docker.com/get-started/)  to be installed onto your machine. I recommend having some knowledge of Docker if you're at all interested in software development, so this could be a good introduction to that tool. This front-end is a popular choice for users of Ollama: https://github.com/open-webui/open-webui