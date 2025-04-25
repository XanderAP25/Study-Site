---
layout: default
title: Case Study
---

# Case Study

After my fact finding expedition, it was time to put my newfound knowledge to the test. Many of my readings touched on the capability of LLMs to generate poetry, and while I thought that was interesting, I am not much of a poetry guy. Instead, I wanted to see how an LLM performed in the generation of German Fairy Tales under three different constraints: a prompt telling an unmodified LLM to create a German Fairy Tale with no examples, a prompt telling it to create a German Fairy Tale with at least one story from *Grimm's Fairy Tales* as an example, an LLM fine-tuned on *Grimm's Fairy Tales* given the same prompt as the unmodified LLM. I picked German Fairy Tales as the topic of generation due to personal interest, but also because my pre-existing knowledge of the culture and their stories from my German minor would make the eventual text analysis easier.

The following set of headings all link to Google Colab notebooks that contain the code and my own insight into the fine-tuning of an LLM, the generation of stories and prompt engineering that went into it, and my eventual text analysis of the generated stories compared to each other and *Grimm's Fairy Tales*. While these notebooks serve as a record of my own work and research, I hope that they can also help you along in your AI journey as code examples, or just sources of inspiration.

## <a href="https://colab.research.google.com/drive/1ue50VMGv12nzZ6uQNxL6wITtvgJ0nX5V?usp=sharing" target="_blank">Finetuning an LLM on German Fairy Tales</a>

A notebook following the process of downloading an AI model from Hugging Face, obtaining a set of data, processing it into a usable format for fine-tuning, and finally carrying out the fine-tuning and saving the model to my computer.

## <a href = "https://colab.research.google.com/drive/1goVTnNt6FauofB_BAQ2Db4h8uWFvfY1d" target="_blank">Story Generation Using an LLM in a Notebook Environment</a>

A notebook containing the step-by-step process of loading a model from Hugging Face, or your own computer for story generation, and completing said task. It also contains notes on my methodology with the prompt creation

## <a href = "https://colab.research.google.com/drive/1_q8NFdDmb1_QonBbXd9wk84RiPA8ei8l?usp=sharing#scrollTo=BMzdSSBlq8dq" target="_blank">Text Analysis of Output and Grimms</a>

A notebook walking through the analysis of the output of a pretrained LLM with a baseline prompt, ICL prompt, and a fine-tuned LLM compared with a corpus of Grimm's Fairy Tales. The findings evaluate the models for how closely they get to true fairy tales.