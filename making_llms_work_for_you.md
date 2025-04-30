---
layout: default
title: Making LLMs Work For You
---
## Section Four: Making LLMs Work for You

When it comes to making LLMs work for you, there are two methods of doing so. The first that we will cover is accessing LLMs via APIs.

An LLM API lets you connect your computer directly to ultra powerful AI models, like GPT-4 and Llama 2, without actually having to download and run the model on your computer. What this means for you is that you can have the most mediocre hardware on the planet, but as long as you have a wifi connection and the ability to pay the fee to access the model, you have an LLM that you can use and modify to your heart's content. The fee to use these APIs is often not a monthly subscription, but instead a fee per million tokens processed as input and output. For reference, one token is about 4 characters, or about three-fourths of a word. This fee depends on what model you're paying to access, but can be anywhere from \\$30 per million tokens processed as output for OpenAI's GPT-4 Turbo model to \\$2.00 per million tokens processed as output for their smaller GPT-3.5 Turbo model. Prices vary depending on what company's model you use, but there will typically be a price associated with using this larger models.

The cost to use these models presents some concerns when you're just starting out with using LLMs. For example, what constitutes a heavy workload for these models? You could be getting acquainted with your shiny new LLM and suddenly find yourself forking over way more cash than you initially intended because you didn't realize that the multi-hour long conversation or learning session you had with your model required millions of input and output tokens to be processed. There are also privacy concerns associated with using LLM APIs, since they typically store your prompts and outputs to improve their own technology. This data is also used to enforce the company's TOS, prevent illegal activity using their technology, and in specific circumstances can be shared with other parties [^14]. Those circumstances include: at the request of government authorities, to certain service providers and vendors when business needs must be met, and during the diligence process with those assisting in a business transfer[^14]. Since user data is retained, that also opens you up to the possibility of your data being included in a data leak if the company that provides your LLM gets hacked. 

Despite these concerns, LLM APIs are typically at the cutting edge of AI technology. So, if you're looking for the most powerful model for a specific application, using an API might be the right choice.

As mentioned earlier, another option is downloading and running an LLM directly on your own computer. Running LLMs locally gives you more control over your data, as it is stored entirely on your device and never shared with the model's creator. There are also no fees for running a local LLM. While this might sound like the ideal option, there are several caveats to consider. For one, these models require fairly powerful hardware to run efficiently. Although there is no fee to use a local LLM, you will likely need to invest in a powerful PC to get the most out of the technology. Smaller models can be run on more modest hardware, such as a relatively modern laptop, but you'll likely experience slower output and lower quality compared to running top-tier models on a high-performance PC with a strong NVIDIA GPU. (Touch on electricity use and the likelihood of it raising your electricity bill).

**(I should also touch on GPT4ALL and LM Studio here, hugging face API free tier)**

To get an LLM up and running on your own machine, you will need a platform to run them on. One of the most popular platforms for this purpose is [Ollama](https://ollama.com), an open-source tool that allows you to run AI models that you have downloaded through a terminal. Setup is fairly straightforward, just download the application from the website, and follow the steps it gives you to get up and running. The Ollama website also hosts models for you to get up and running, but you can also run models from Hugging Face if there is something there that is more suited to your needs (i just realized that I have not talked about hugging face at all in this entire article). These models are measured by how many parameters they have, and they are formatted like "8B" meaning 8-billion parameters in this case. For weaker computers, like laptops, you will want to use models that are at most 8B. Depending on how powerful your laptop is, you could go higher than that, especially if it has a dedicated NVIDIA GPU. For PCs, you can go higher than this, but will still find a limit around the 30B parameter mark. Despite that being the limit, there are models available for download with upwards of 70B parameters, going into the 600-billions.

To run these models, you would either need to drop a small fortune on a rig dedicated to running massive AI models, or purchase access to a cloud computing service, and use their hardware instead. In a way, this is the middleground between running an LLM on your own computer and using an API to access a company's AI model. You have some of the benefits of running an LLM on your machine, like keeping your data private and the ability to pick and choose specific models you want to use on the fly. And some benefits from LLM APIs, like gaining access to an extremely powerful LLM while not needing a powerful computer. You do have to pay to use these cloud services, but these rates are hourly, instead of usage-based, which could be good or bad depending on what you're doing with your LLM.

**(There is a way to properly end this section, but as a first draft, I think this is a good spot to build from after critique)**

<p style="text-align:left;">
    <a href="{{ '/training_and_fine_tuning_an_llm.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">← Training and Fine-Tuning an LLM</a>
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