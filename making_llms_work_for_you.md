---
layout: default
title: Making LLMs Work for You
---

## Section Three: Making LLMs Work for You

When it comes to making LLMs work for you, there are two main methods. The first we will cover is accessing LLMs via APIs.

An LLM API lets you connect your computer directly to ultra-powerful AI models, like GPT-4 and Llama 2, without having to download and run the model on your own machine. What this means for you is that regardless of your hardware, as long as you have a Wi-Fi connection and the ability to pay the fee to access the model, you have an LLM that you can use and modify to your heart's content. The fee to use these APIs is often not a monthly subscription but instead a fee per million tokens processed as input and output. This practice has become the standard across most AI companies, but some AIs, like Mistral, offer monthly subscriptions. For reference, one token is about four characters, or roughly three-fourths of a word. This fee depends on the model you're paying to access but can range from \$30 per million tokens processed as output for OpenAI's GPT-4 Turbo model to \$2.00 per million tokens processed as output for their smaller GPT-3.5 Turbo model. Prices vary depending on the company and model, but there will typically be a price associated with using these larger models.

The cost of using these models presents some concerns when you're just starting out. For example, what constitutes a heavy workload for these models? You could be getting acquainted with your shiny new LLM and suddenly find yourself paying way more than you initially intended because you didn’t realize that the multi-hour conversation or learning session you had required millions of input and output tokens to be processed. There are also privacy concerns associated with using LLM APIs, as they typically store your prompts and outputs to improve their technology. This data is also used to enforce the company’s TOS, prevent illegal activity, and, in specific circumstances, may be shared with other parties ([^14]). Those circumstances include requests from government authorities, disclosures to certain service providers and vendors when business needs must be met, and during diligence processes involving business transfers ([^14]). Since user data is retained, this also raises the risk of your data being exposed in a data breach if the company that provides your LLM is hacked.

Despite these concerns, LLM APIs are typically at the cutting edge of AI technology. So if you're looking for the most powerful model for a specific application, using an API might be the right choice.

If relying on an external API doesn't suit your needs or priorities, an alternative approach is to run an LLM directly on your own computer. Running LLMs locally gives you more control over your data, as it remains entirely on your device and is never shared with the model's creator. There are also no fees for running a local LLM, aside from a possible increase in your electricity bill if you run them for extended periods. While this might sound ideal, there are several caveats to consider. For one, these models require fairly powerful hardware to run efficiently. Although there is no fee to use a local LLM, you will likely need to invest in a powerful PC to get the most out of the technology. Smaller models can run on more modest hardware, such as a relatively modern laptop, but you'll likely experience slower output and lower quality compared to running top-tier models on a high-performance PC with a strong NVIDIA GPU. There are also smaller quantized versions of larger LLMs that offer high-quality output while being able to run on less powerful hardware ([^19]).

To get an LLM up and running on your own machine, you will need a platform to run it. One of the most popular platforms for this purpose is [Ollama](https://ollama.com), an open-source tool that allows you to run AI models that you have downloaded through a terminal. Setup is fairly straightforward—just download the application from the website and follow the steps to get up and running. The Ollama website hosts models for immediate use, but you can also run community models from Hugging Face or custom models of your own design if there is a specific LLM better suited to your needs. Other than Ollama, [LM Studio](https://lmstudio.ai) is a similar application that provides a graphical interface for interacting with your AI models instead of a terminal-based approach like Ollama. There are also web-based frontends for Ollama that you can set up, but they require some technical knowledge, which I will not be covering here ([^20]).

Another popular option for running local models is [GPT4All](https://www.nomic.ai/gpt4all). It stands out for its user-friendliness, offering a clean, ChatGPT-like interface. Unlike platforms like Ollama, which rely on terminal commands, GPT4All is more plug-and-play. Once installed, it allows you to easily browse and download models directly within the application, requiring no additional repositories or command-line interactions. GPT4All also lists the RAM requirements for its models, which is incredibly helpful when deciding which model to use.

AI models are measured by the number of parameters they have, formatted like "8B," meaning 8 billion parameters in this case. For everyday computers like laptops, you will want to use models that are at most 8B. Depending on how powerful your laptop is, you could go higher than that, especially if it has a dedicated NVIDIA GPU. For desktop PCs with powerful GPUs, you can go higher but will typically find a limit around the 30B parameter mark. Despite that, there are models available for download with upwards of 70B parameters, extending into the hundreds of billions.

To run those massive models, you would either need to invest in a high-end rig dedicated to AI processing or purchase access to a cloud computing service and use their hardware instead. In a way, this is the middle ground between running an LLM on your own computer and using an API to access a company's AI model. You gain some of the benefits of local LLMs, such as data privacy and model flexibility, while also enjoying the power and scalability of cloud-based models. You do have to pay to use these cloud services, but the rates are hourly instead of usage-based, which could be good or bad depending on your workflow.

In my case, where I wanted to not only generate stories but also fine-tune a model (which we will cover in the next section), I opted to purchase compute units from Google to use with their Google Colab notebook environment. This granted me access to an incredibly powerful GPU, allowing me to generate stories at a far faster rate than on my local machine while also enabling easy sharing of my code since it was hosted online. For most introductory AI applications, I would recommend Google Colab, as there is a free tier that allows experimentation with more powerful models than most consumer-grade machines can handle.

Ultimately, your choice of interface and machine when using an LLM will be based on your needs and preferences. If you just want a private chatbot that lacks the usage limits of free-tier AI models like ChatGPT, then Ollama, LM Studio, or GPT4All will likely suit your needs, assuming your machine can run smaller or quantized models. If you are looking into tuning or training your own LLM or using a top-of-the-line LLM with no compromises, then an API or cloud computing solution will be more suited to your workflow. Just ensure you do your due diligence in selecting an API or solution that best fits your needs.

<p style="text-align:left;">
    <a href="{{ '/how_llms_can_benefit_you_and_their_challenges.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">← How LLMs Can Benefit You and Their Challenges</a>
    <span style="float:right;">
        <a href="{{ '/training_and_fine_tuning_an_llm.html' | relative_url }}" style="padding: 0.4em 0.8em; border: 1px solid #1e6bb8; color: #1e6bb8; text-decoration: none; border-radius: 3px; font-weight: bold;">Training and Fine-Tuning an LLM →</a>
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

[^18]: LLMs use tons of VRAM, and require powerful GPUs to be used to their fullest extent. Whether that be output generation, or fine-tuning, you want to have a lot of VRAM at your disposal. You can get away with using CPUs or Apple Silicon in conjunction with your system RAM for running and fine-tuning smaller LLMs, but you will see noticeably less performance and efficiency. Google Colab is a relatively cheap resource to rent powerful GPUs that you can easily use for your AI ambitions. You can even use a free T4 GPU, which offers more power than what the layman would likely have on hand. A good rule of thumb is that you want at least 16 GB of VRAM to do anything significant with moderate sized LLMs (1-3B parameters), and you may want 80GB+ of VRAM if you're fine-tuning larger models.

[^19]: Quantized models have their weights reduced in size from (typically) 32-bit floats to 8-bit integers. This significantly reduces their size and computational overhead while making them less precise in their outputs, meaning that they will have somewhat lower quality generated text. These models still benefit from large training sets and expert optimization, however. For further reading, see Hugging Face’s documentation on quantization: https://huggingface.co/docs/optimum/en/concept_guides/quantization

[^20]: Any web-based frontend for Ollama will require [Docker](https://docs.docker.com/get-started/)  to be installed onto your machine. I recommend having some knowledge of Docker if you're at all interested in software development, so this could be a good introduction to that tool. This front-end is a popular choice for users of Ollama: https://github.com/open-webui/open-webui