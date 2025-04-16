---
layout: default
title: Notebook
---

# [Text Analysis of Outputs of LLMs Compared to Grimm's Fairy Tales](https://colab.research.google.com/drive/1_q8NFdDmb1_QonBbXd9wk84RiPA8ei8l?usp=sharing)


```python
# Clone the fighting words algorithm
!git clone https://github.com/jmhessel/FightingWords.git
```

    fatal: destination path 'FightingWords' already exists and is not an empty directory.



```python
# Load packages
import pandas as pd
import numpy as np
import os
import sklearn.feature_extraction.text as sk_text
from FightingWords import fighting_words_py3 as fighting_words
import seaborn as sns
import matplotlib.pyplot as plt

# For word features
import spacy

import warnings
warnings.filterwarnings("ignore")
```


```python
# Load in the data
grimms_fairy_tales_df = pd.read_csv("data/grimms_fairy_tales.csv")
baseline_stories_df = pd.read_csv("data/baseline_stories_hightemp_output.csv")
ICL_stories_df = pd.read_csv("data/ICL_stories_output.csv")
finetuned_stories_df = pd.read_csv("data/finetune_stories_output.csv")
```


```python
# Combine all items in the output/Text column into a single corpus
baseline_stories_output_corpus = baseline_stories_df.output.to_list()
ICL_stories_output_corpus = ICL_stories_df.output.to_list()
grimms_fairy_tales_corpus = grimms_fairy_tales_df.Text.to_list()
finetuned_stories_output_corpus = finetuned_stories_df.output.to_list()
```

## Fighting Word Analysis


```python
# Fighting Words Functions

print("Baseline Pretrained Model: ")
baseline_fighting_words = fighting_words.bayes_compare_language(baseline_stories_output_corpus, grimms_fairy_tales_corpus)
print("\nPretrained Model Using In-Context Learning: ")
ICL_fighting_words = fighting_words.bayes_compare_language(ICL_stories_output_corpus, grimms_fairy_tales_corpus)
print("\nFinetuned Model: ")
finetuned_fighting_words = fighting_words.bayes_compare_language(finetuned_stories_output_corpus, grimms_fairy_tales_corpus)

# Convert fightin words to dataframes
baseline_FW_df = pd.DataFrame(baseline_fighting_words, columns=['word', 'z-score']).assign(type="baseline").sort_values(by='z-score', ascending=False)
ICL_FW_df = pd.DataFrame(ICL_fighting_words, columns=['word', 'z-score']).assign(type="ICL").sort_values(by='z-score', ascending=False)
finetune_FW_df = pd.DataFrame(finetuned_fighting_words, columns=['word', 'z-score']).assign(type="fine-tune").sort_values(by='z-score', ascending=False)
```

    Baseline Pretrained Model: 
    Vocab size is 873
    Comparing language...
    
    Pretrained Model Using In-Context Learning: 
    Vocab size is 808
    Comparing language...
    
    Finetuned Model: 
    Vocab size is 746
    Comparing language...


The criteria for a word to be included in the vocabulary list of the fighting words algorithm are that it must:
 - appear in either of the two corpora.
 - appear at least 10 times.
 - appear in no more than 50% of the stories.

We can see that the vocabulary list decreases in size as we go from using a baseline pretrained model, to a pretrained model with some examples, to a model that is fine-tuned on Grimm's Fairy Tales. Two possibilities come to mind here, the first being that as the model is being given a more solid idea of what a "German Fairytale" is, the word choice of the output could be getting closer to the original Grimm's Fairy Tales corpus. Due to this, we are seeing a greater number of words that appear in more than 50% of the stories and getting a smaller vocabulary list as a result. It is also possible that this decreasing vocabulary size is a result of the outputs becoming less varied in their word choice as they are given examples or trained on the corpus.

### Baseline Pretrained Model Fighting Words Visualization


```python
# Let's look at the top and bottom 15 words of baseline
baseline_top = baseline_FW_df.head(15)
baseline_bottom = baseline_FW_df.tail(15)
baseline_top_and_bottom = pd.concat([baseline_top, baseline_bottom])

# Create a 'type' column where the positive numbers are 'Baseline' and the negative numbers are 'Grimms'
baseline_top_and_bottom['type'] = np.where(baseline_top_and_bottom['z-score'] > 0, 'Baseline', 'Grimms')

# Plot
plt.figure(figsize=(6, 6))
sns.barplot(data=baseline_top_and_bottom, y='word', x='z-score', hue='type',palette="colorblind", dodge=False)
plt.title("15 Most Distinct Words Between Stories from a Pretrained Model and Grimm's Fairy Tales")
plt.show()
```


    
![png](assets\images\text_analysis_notebook_9_0.png)
    


It is immediately clear that "Hans" is an overwhelmingly distinct word that appeared in the baseline output. Hans is a relatively popular name in Germany. So, while it is surprising to see it featured so prominently, it can be explained as the model just making the connection between Hans and "German Fairy Tales" and using this stereotypical name as a crutch when a character needed a name. Grimm's Fairy Tales see more variety in their names, and while Hans is used in the story *Hans in Luck* it isn't featured much aside from that. Instead, we get more unique names like "Chanticleer" and "Partlet", or the story just referring to characters by what they are, like "the king," which just so happens to be the most distinct word in the Grimm's corpus. We also see this with words like "mother" and "father" featuring in the top 15, which could align with the idea of calling characters what they are. More likely, this is likely a result of German fairy tales frequently featuring themes or lessons for children to follow, with one of the more popular ones being listenting to your parents.

The baseline output is comparatively more fantastical with its distinct words, with "witch", "spirit", "golden", and "wish" featuring as some of the most distinct words. Fairy tales are fantasy stories, there are elements of the supernatural in them, and the model appears to be generating stories with these fantastical elements, but with less emphasis on the themes of family that feature so prominently in the actual stories. We also see that "love" is featured more prominently in the generated stories, meaning that the stories could be more fantastical stories with elements of love compared to the supernatural stories with advisory messages for children.

Finally, we see that "village" and "trees" are featuring heavily in the generated output. Villages and forests are prominent locations in the generated outputs, and often in Grimm's Fairy Tales. For it to be distinct to such a degree on the side of the generated output is interesting. Grimm's Fairy Tales have a larger variety of locations, and often have named locations too, so it is possible that there is a lack of a defined name for many of the locations in the generated output, resulting in the distinctiveness of "village" and "trees".

The model emulates the aesthetics of a fairy tales with its use of words and language hinting at fantastical elements and  settings based in villages and forests, but fails to achieve the same moral and familial themes that are associated with the originals.

### ICL Model Fighting Words Visualization


```python
# Let's look at the top and bottom 15 words of ICL
ICL_top = ICL_FW_df.head(15)
ICL_bottom = ICL_FW_df.tail(15)
ICL_top_and_bottom = pd.concat([ICL_top, ICL_bottom])

# Create a 'type' column where the positive numbers are 'ICL' and the negative numbers are 'Grimms'
ICL_top_and_bottom['type'] = np.where(ICL_top_and_bottom['z-score'] > 0, 'ICL', 'Grimms')

# Plot
plt.figure(figsize=(6, 6))
sns.barplot(data=ICL_top_and_bottom, y='word', x='z-score', hue='type',palette="colorblind", dodge=False)
plt.title("15 Most Distinct Words Between Stories from a Pretrained Model Using ICL and Grimm's Fairy Tales")
plt.show()
```


    
![png](assets\images\text_analysis_notebook_12_0.png)
    


Hans still features prominently, but has fallen from a z-score of 26.6 to around 7.41, and sits below "fox" and "bread". Those words, along with "loaf" indicate a more fable-like quality to the output, with Aesop in particular coming to mind. Foxes feature prominently in some actual Germanic fairy tales, so it is possible that when given an example with a fox, or bread, the model latched onto it as prime fairy tale material.

There appears to be more of an overlap here with themes than in the previous graph. Royalty is distinctly featured in both corpora, with "prince" and "queen" being featured in the generated output and Grimm's respectively. "Bride" being prominent alongside "queen" indicates that while royalty features in both corpora, their roles may differ somewhat and could be indicative of some gendered bias in the traditional fairy tales that might be missing in the modern language model.

The distinct words in the generated output still align more with stereotypical Germanic storytelling elements, like "village" and "forest" being common settings for the story. We also see more words used to assign a name to something like "named" or "called" which goes hand in hand with Hans featuring so heavily. What makes this significant is that it is much closer to the way that a character would be named in Grimm's Fairy Tales, showing that the model is successfully using the examples in some form to get closer to an actual fairy tale.

Interestingly, Grimm's stories might be darker, with "dead" being the third most distinct word in that corpus, compared to "danger" and "returned" in the generated output. What this could mean is that less characters see a happy ending in the actual fairy tales compared to the generated ones, where characters might face some danger or challenge, but ultimately come back alive. The distinctiveness of "thee" also stands out as an indication that the generated outputs may be showing more contempoary dialogue or prose compared to the older Grimm's Fairy Tales, despite having examples to draw from and being instructed to write a traditional fairy tale.

With an ICL prompt providing one to a few examples, the output is now incorporating language and themes that are closer to the original. The manner of designating a character using "named" or "called" in particular is much closer to a proper fairy tale. Royalty also seems to feature more prominently now, even if the exact royal being featured is different and likely an indication of a difference in gendered bias. While the language is closer to the original, it doesn't seem to be entirely there, with the generated output being more modern in prose and dialogue. The generated stories also seem to avoid grisly ends for the characters, instead more commonly ending in them returning from whatever danger they faced.

### Fine-Tuned Model Fighting Words Visualization


```python
# Let's look at the top and bottom 15 words of Finetune
finetune_top = finetune_FW_df.head(15)
finetune_bottom = finetune_FW_df.tail(15)
finetune_top_and_bottom = pd.concat([finetune_top, finetune_bottom])

# Create a 'type' column where the positive numbers are 'fine-tune' and the negative numbers are 'Grimms'
finetune_top_and_bottom['type'] = np.where(finetune_top_and_bottom['z-score'] > 0, 'Fine-tune', 'Grimms')

# Plot
plt.figure(figsize=(6, 6))
sns.barplot(data=finetune_top_and_bottom, y='word', x='z-score', hue='type',palette="colorblind", dodge=False)
plt.title("15 Most Distinct Words Between Stories from a Finetuned Model and Grimm's Fairy Tales")
plt.show()
```


    
![png](assets\images\text_analysis_notebook_15_0.png)
    


Post-finetune, Hans is no longer in the top 15 most distinct words, and this is likely due to the model being tuned on more names that are actually used in German fairy tales. "Queen" now also features more distinctly in the generated output than in the Grimm's corpus, meaning that the generated outputs could be seeing more focus on royalty relative to Grimm's fairy tales. Interestingly, the distinct words of the fine-tuned output are more grounded than the baseline model's output, with the only fantasy-adjacent word being "dwarf" which is a fantasy creature closely related with Germanic folklore. We have occupations like "miller", "woodsman", and "shepherd" featuring more distinctly in the generated output. These could be common origins for characters before they go through their transformative adventure or ordeal. This is aligns with the structure of German fairy tales, and while the previous models implied some humble beginnings due to the prominence of "village," the added variety shows a higher level of understanding to what a Germanic humble beginning could look like.

"Snow" is also a new distinct word, and this likely coincides with "queen", as the "Snow Queen" was a character that was featured prominently in multiple generated stories. What is fascinating about this is that the *The Snow Queen* is a Danish fairy tale that was written around the time when Grimm's Fairy Tales was being published and updated. *The Snow Queen* wasn't used in the training set for the fine-tuned model, yet the model is still showing a preference for this specific character post-tuning. It could be that this "snow queen" is a common archetype that represents a strong woman or force in the story that the model draws on as needed, similar to Hans with the baseline outputs. This figure could symbolize a powerful female presence, an evolution from the more traditional and often passive roles like "bride" and "lady" that were more distinct the Grimm's corpus. In this waym the model appears to go beyond its fine-tuning data to create a more modern character archetype than simply regurgitating older, more submissive portrayaks of women in fairy tales.

### Word Features


```python
nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    features = {
        "avg_word_length": np.mean([len(w) for w in words]), # Word length
        "type_token_ratio": len(set(words)) / len(words), # Diversity of word choice in text (unique words/total words)
        "avg_sentence_length": np.mean([len(sent.text.split()) for sent in sentences]), # Sentence length
        "noun_ratio": sum(1 for token in doc if token.pos_ == "NOUN") / len(words), # Ratio of nouns to length of text
        "adj_ratio": sum(1 for token in doc if token.pos_ == "ADJ") / len(words), # Ratio of adjectives to length of text
        "pronoun_ratio": sum(1 for token in doc if token.pos_ == "PRON") / len(words), # Ratio of pronouns to length of text
    }
    return features

baseline_features = [extract_features(text) for text in baseline_stories_output_corpus]
ICL_features = [extract_features(text) for text in ICL_stories_output_corpus]
finetune_features = [extract_features(text) for text in finetuned_stories_output_corpus]
grimms_fairy_tales_features = [extract_features(text) for text in grimms_fairy_tales_corpus]
```


```python
word_features = baseline_features + ICL_features + finetune_features + grimms_fairy_tales_features
word_features_df = pd.DataFrame(word_features)
word_features_df['model'] = ['baseline'] * len(baseline_features) + ['ICL'] * len(ICL_features) + ['fine-tune'] * len(finetune_features) + ['Grimms'] * len(grimms_fairy_tales_features)
word_features_df
```





  <div id="df-c1efedb5-58ad-4795-a984-bbec7c267408" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_word_length</th>
      <th>type_token_ratio</th>
      <th>avg_sentence_length</th>
      <th>noun_ratio</th>
      <th>adj_ratio</th>
      <th>pronoun_ratio</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.146862</td>
      <td>0.432577</td>
      <td>13.690909</td>
      <td>0.220294</td>
      <td>0.057410</td>
      <td>0.160214</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.534783</td>
      <td>0.439130</td>
      <td>14.978261</td>
      <td>0.257971</td>
      <td>0.066667</td>
      <td>0.073913</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.423272</td>
      <td>0.433390</td>
      <td>14.512195</td>
      <td>0.205734</td>
      <td>0.079258</td>
      <td>0.111298</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.328407</td>
      <td>0.459770</td>
      <td>16.263158</td>
      <td>0.218391</td>
      <td>0.078818</td>
      <td>0.082102</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.207337</td>
      <td>0.438596</td>
      <td>14.952381</td>
      <td>0.215311</td>
      <td>0.060606</td>
      <td>0.135566</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>247</th>
      <td>3.855374</td>
      <td>0.215753</td>
      <td>17.500000</td>
      <td>0.136196</td>
      <td>0.046365</td>
      <td>0.177555</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>248</th>
      <td>3.822461</td>
      <td>0.322798</td>
      <td>19.851351</td>
      <td>0.147276</td>
      <td>0.053800</td>
      <td>0.179556</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>249</th>
      <td>3.892580</td>
      <td>0.230363</td>
      <td>21.386667</td>
      <td>0.166408</td>
      <td>0.045017</td>
      <td>0.166408</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>250</th>
      <td>3.746387</td>
      <td>0.229371</td>
      <td>28.716216</td>
      <td>0.171562</td>
      <td>0.044755</td>
      <td>0.144988</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>251</th>
      <td>4.044068</td>
      <td>0.302072</td>
      <td>23.700000</td>
      <td>0.181544</td>
      <td>0.070810</td>
      <td>0.131827</td>
      <td>Grimms</td>
    </tr>
  </tbody>
</table>
<p>252 rows × 7 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c1efedb5-58ad-4795-a984-bbec7c267408')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c1efedb5-58ad-4795-a984-bbec7c267408 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c1efedb5-58ad-4795-a984-bbec7c267408');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-645f0a52-c35b-4cba-926f-f10a5bb022c4">
  <button class="colab-df-quickchart" onclick="quickchart('df-645f0a52-c35b-4cba-926f-f10a5bb022c4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-645f0a52-c35b-4cba-926f-f10a5bb022c4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_3138ffec-3d53-4851-9ee3-4fb17f6ac237">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('word_features_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_3138ffec-3d53-4851-9ee3-4fb17f6ac237 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('word_features_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Plot the word features
fig, axes = plt.subplots(2, 3, figsize=(10,7))
fig.suptitle('Word Features by Model', fontsize=16)

x = 0
y = 0
for i in word_features_df.columns:
    if i != 'model':
        sns.boxplot(x='model', y=i, hue='model', data=word_features_df, ax=axes[x, y], notch=True)
        axes[x, y].set_title(f"{i} by model")
        y += 1
        if y == 3:
            y = 0
            x += 1
plt.tight_layout()
plt.show()
```


    
![png](assets\images\text_analysis_notebook_20_0.png)
    


From our word features analysis, we can see that using either in-context learning or fine-tuning generally brings the structure and word choice of the output closer in line with Grimm's Fairy Tales. This isn't too surprising, since ICL gives the model a few examples to work from, while fine-tuning exposes it to the entire corpus. What's more interesting is just how close the ICL and fine-tuned outputs are to each other, and to Grimm's, across most metrics. For the most part, the fine-tuned outputs are slightly closer to Grimm's across the board, but the gap isn't always that big. In many cases, ICL gets close enough that it raises the question of whether full fine-tuning is actually necessary if the goal is just stylistic mimicry.

The biggest drawback to using the fine-tuned model was the time it took to generate the stories. With ICL, we were able to generate 63 stories in about 45 minutes. The fine-tuned model took around 3.5 hours to produce the same number. Fine-tuning itself only took 8 minutes. The time to generate makes a big difference if you're working under time or budget constraints. A model of this size (7B parameters) needs access to high-end GPUs—in this case, an A100 rented using Google Cloud compute credits, which you have to purchase, and are deducted hourly. If the goal is to generate a large number of stories that follow the style of German folktales, the fine-tuned model will likely give you slightly better outputs, but at a higher cost. In the time it takes to generate 63 stories with the fine-tuned model, you could generate over 200 using ICL with the pretrained model. It really comes down to your use-case. Do you want a smaller number of stories closer in style to actual German fairy tales, or a larger batch of stories that are are not as close to German fairy tales, but much cheaper and faster to generate?

### Gendered Pronoun Analysis

We saw in the fighting words analysis that there might be a difference in portrayal of genders between the traditional German fairy tales and the outputs generated by an AI model with modern biases. As a final point of analysis, let's examine the models for use of gendered pronouns to see if one gender is featured more than the other, and what this could mean for a pretrained model taking examples from, or being tuned, on data with more archaic gender roles.


```python
from collections import Counter

def count_gender_by_type(text):
    doc = nlp(text)
    masculine = {"he", "him", "his", "himself"}
    feminine = {"she", "her", "hers", "herself"}

    gender_counts = {"masculine": 0, "feminine": 0, "total": 0}
    for token in doc:
        word = token.text.lower()
        if word in masculine:
            gender_counts["masculine"] += 1
            gender_counts["total"] += 1
        elif word in feminine:
            gender_counts["feminine"] += 1
            gender_counts["total"] += 1
    #print(gender_counts['total'])
    gender_ratio = {"masculine": gender_counts["masculine"] / gender_counts["total"], "feminine": gender_counts["feminine"] / gender_counts["total"]}
    return gender_ratio


baseline_pronouns = [count_gender_by_type(text) for text in baseline_stories_output_corpus]
ICL_pronouns = [count_gender_by_type(text) for text in ICL_stories_output_corpus]
finetune_pronouns = [count_gender_by_type(text) for text in finetuned_stories_output_corpus]
grimms_fairy_tales_pronouns = [count_gender_by_type(text) for text in grimms_fairy_tales_corpus]
```


```python
gendered_pronouns = baseline_pronouns + ICL_pronouns + finetune_pronouns + grimms_fairy_tales_pronouns
gendered_pronouns_df = pd.DataFrame(gendered_pronouns)
gendered_pronouns_df['model'] = ['baseline'] * len(baseline_pronouns) + ['ICL'] * len(ICL_pronouns) + ['fine-tune'] * len(finetune_pronouns) + ['Grimms'] * len(grimms_fairy_tales_pronouns)
gendered_pronouns_df
```





  <div id="df-99626d0f-58a9-4639-91b4-b5ff5da78ad6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>masculine</th>
      <th>feminine</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.758065</td>
      <td>0.241935</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.937500</td>
      <td>0.062500</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.600000</td>
      <td>0.400000</td>
      <td>baseline</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>247</th>
      <td>0.953782</td>
      <td>0.046218</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.317757</td>
      <td>0.682243</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.932489</td>
      <td>0.067511</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.280488</td>
      <td>0.719512</td>
      <td>Grimms</td>
    </tr>
    <tr>
      <th>251</th>
      <td>0.747253</td>
      <td>0.252747</td>
      <td>Grimms</td>
    </tr>
  </tbody>
</table>
<p>252 rows × 3 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-99626d0f-58a9-4639-91b4-b5ff5da78ad6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-99626d0f-58a9-4639-91b4-b5ff5da78ad6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-99626d0f-58a9-4639-91b4-b5ff5da78ad6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ee784370-6abc-4d26-9b60-ec43f09f699a">
  <button class="colab-df-quickchart" onclick="quickchart('df-ee784370-6abc-4d26-9b60-ec43f09f699a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ee784370-6abc-4d26-9b60-ec43f09f699a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_da23120a-87a3-44bc-b6ad-2453f9ab23dd">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('gendered_pronouns_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_da23120a-87a3-44bc-b6ad-2453f9ab23dd button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('gendered_pronouns_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Plot the gendered pronoun boxplot
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle('Gendered Pronouns by Model', fontsize=16)
j = 0
for i in gendered_pronouns_df.columns:
    if i != 'model':
        sns.boxplot(x='model', y=i, hue='model', data=gendered_pronouns_df, ax=axes[j], notch=True)
        axes[j].set_title(f"{i} pronoun ratio by corpus")
        j += 1
plt.tight_layout()
plt.show()
```


    
![png](assets\images\text_analysis_notebook_26_0.png)
    


Across all stories, male pronouns are far more common, which makes sense given what we saw earlier in the fighting words analysis of the outputs compared to Grimm's fairy tales. The prominence of Hans, and traditionally male professions like woodsman, millers, and shepherd likely have a role to play in the larger representation of men in the generated stories. It seems that there is overlap between all the notches of the boxes in the masculine box plot, meaning that there is not a statistically significant enough change in male pronoun use to conclude that there has been a downward shift in male representation. We can observe that the first quartile is smaller, and the second quarter goes lower in the ICL, fine-tuned, and Grimms box plots than the baseline, however. This suggests that a larger portion of those stories are using fewer male pronouns, even if the medians are the same. This isn't enough to say that male representation is declining in the ICL and fine-tuned stories, but it does point to a wider range of how prevelant masculine-centered language is used compared to baseline.

The feminine boxplot has notches that all overlap, meaning that there is not a significant change in average female pronoun use. At the same time, the third quartile for Grimms, ICL, and fine-tuned female pronoun ratios are all higher than baseline, and the third quarter's range also caps out at greater ratios. This implies that the stories have a greater amount of female representation compared to baseline, even if their medians are not that different. This could mean that the inclusion of examples or fine-tuning enhances the flexibility of the model in character choice and narrative focus of stories.

## Conclusion

This analysis demonstrates that when large language models are either fine-tuned on or guided with examples from German fairy tales, their generated stories more closely mimic the original Grimm corpus in terms of style, diction, and thematic elements than a pretrained model using no examples or fine-tuning. Both the fighting words and word feature analyses support this conclusion. While the gendered pronoun ratio analysis was less conclusive statistically, visual comparisons suggest that the ICL and fine-tuned outputs more closely resemble the patterns found in Grimm's Fairy Tales than those produced by the baseline model.

Notably, the fine-tuned model consistently outperforms the ICL-based model in most word features metrics, though the margin is narrower than one might expect. This suggests that while fine-tuning offers a more robust transformation of the model's output behavior, ICL remains a viable alternative when resources such as time or compute are constrained. However, this doesn't mean that the fine-tuning process itself could be optimized further. In particular, through expansion of the training dataset beyond the 63 Grimm stories currently used.

The gendered pronoun analysis also provides some insight into how representation shifts between model outputs. While male pronouns remain dominant across all corpora, there is evidence of more varied representation in the fine-tuned and ICL outputs. Female pronouns, while not significantly more frequent on average, appear in a wider range of proportions, indicating greater flexibility in narrative focus.

These generated stories show the potential for LLMs to stylistically adapt when exposed to domain-specific data, and highlights the trade-offs between fine-tuning and ICL. Future work could be done via taking a deeper analysis of the generated stories by looking at their syntactic complexity or taking a proper look at how the stories resolve, instead of drawing conclusions based on their distinct words. Redoing the fine-tune to include more stories could also be an interesting starting point for a comparative study.
