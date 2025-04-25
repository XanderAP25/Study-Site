---
layout: default
title: How LLMs Benefit You and Their Challenges
---

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

<p style="text-align:left;">
    <a href="{{ '/intro_to_ai_and_llms.html' | relative_url }}">← Introduction to AI and LLMs</a>
    <span style="float:right;">
        <a href="{{ '/training_and_fine_tuning_an_llm.html' | relative_url }}">Training and Fine-Tuning an LLM →</a>
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