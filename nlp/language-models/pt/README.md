# General Domain Portuguese Language Model

This is an on-going language model implementation for Portuguese (Wikipedia based corpus).  

- AWD-LSTM based architecture <a href="#DBLP58journals47corr47abs4517084502182">[3]</a>
- Vocabulary size: 30,000 words
- Based on <a href="http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html">fast.ai ULMFit approach</a> <a href="#DBLP58journals47corr47abs4518014506146">[1]</a>

## Files

- PT_Language_Model.ipynb: Jupyter notebook with data preparation and several experiments.  The data preparation are strongly inspired by Telugu Language Model <a href="https://github.com/binga/fastai_notes/tree/master/experiments/notebooks/lang_models">[4]</a>.
- PT_Language_Model-1Cycle-Tests.ipynb / PT_Language_Model-1Cycle-Tests.py: Scripts for training for 10 epochs a pretrained (for 2 epochs) model based on 1cycle <a href="#DBLP58journals47corr47abs4518034509820">[2]</a>.  Python scripts (.py) are easier to run due to difficulties with long execution times in Jupyter notebooks. 
- PT_Language_Model-1Cycle-Tests-From-Scratch.ipynb / PT_Language_Model-1Cycle-Tests-From-Scratch.py: Scripts for running for 10 epochs a model from scratch based on 1cycle <a href="#DBLP58journals47corr47abs4518034509820">[2]</a>.  Python scripts (.py) are easier to run due to difficulties with long execution times in Jupyter notebooks.  PS.: I created these scripts after not seeing advantage in running for 10 epochs from a pretrained model, as suggested by T_Language_Model-1Cycle-Tests.ipynb / PT_Language_Model-1Cycle-Tests.py:
- Relatorio de treinamento language model.docx (in Portuguese): Report with detailed information and plots about the training steps.
- Modelo de Linguagem para Português com ULMFit.pdf (in Portuguese): Powerpoint Report

## Instructions to download and pre-process Wikipedia

(Thanks to Phani Srikanth, for the tips cited on <a href="https://github.com/binga/fastai_notes/tree/master/experiments/notebooks/lang_models">[4]</a>)

1. wget https://dumps.wikimedia.org/ptwiki/20180501/ptwiki-20180501-pages-articles-multistream.xml.bz2 
2. wget https://github.com/attardi/wikiextractor/raw/master/WikiExtractor.py
3. bzip2 -d ptwiki-20180501-pages-articles-multistream.xml.bz2
4. python WikiExtractor.py -o data/ ptwiki-20180501-pages-articles-multistream.xml -s --json

## References
1. <a id="DBLP58journals47corr47abs4518014506146"></a>Jeremy Howard and Sebastian Ruder. Fine-tuned Language Models for Text Classification. 2018
2. <a id="DBLP58journals47corr47abs4518034509820"></a>Leslie N. Smith. A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay. 2018
3. <a id="DBLP58journals47corr47abs4517084502182"></a>Stephen Merity, Nitish Shirish Keskar and Richard Socher. Regularizing and Optimizing LSTM Language Models. 2017
4. <a href="https://github.com/binga/fastai_notes/tree/master/experiments/notebooks/lang_models">Language Model for Telugu (Indian) Language</a>
5. For useful tips about "1cycle" (including suggested hyperparameters), please see sgugger's posts <a href="http://forums.fast.ai/t/language-model-zoo-gorilla/14623/49">here</a>
