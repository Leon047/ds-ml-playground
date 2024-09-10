# Efficient Causal Graph Discovery Using Large Language Models.

### Causal graph
<div align="center" style="width: 40%">
    <image src="causal_graph.png">
</div>

## Description
* [EFFICIENT CAUSAL GRAPH DISCOVERY USING LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2402.01207)

Algorithm BFS with LLMs:
<pre>
Require: LM pθ, descriptions of variables X, initial variable selector I(), expansion generator E(), cycle checker CheckCycle()
G ← {} ▷ Create an empty graph to store the result.
frontier, visited ← I(pθ, X) ▷ With initialization prompt.
while frontier is not empty do
    toV isit ← frontier [0]
    frontier.remove(toV isit)
    visited.add(toV isit)
    for node in E(pθ, G) do ▷ Expand with expansion prompt.
        if not CheckCycle(G, toV isit, node) then ▷ Check if adding toV isit → node will create cycle.
            G.add((toV isit, node))
        end if
        if node not in frontier ∪ visited then
            frontier.add(node)
        end if
    end for
end while
return G
</pre>

## Installation
In the project directory.

### 1. Download `Dataset.csv`: 
* [Prediction Of Sepsis Dataset](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

### 2. Create a `.env` file and add content:
```bash
# OpenAI api key          
OPENAI_API_KEY='sk-'
```

### 3. Install dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### 4. Run the project:
```bash
python run.py
```
