"""
Efficient Causal Graph Discovery Using Large Language Model.

Docs:
* Efficient Causal Graph Discovery: https://arxiv.org/abs/2402.01207
* OpenAI_API: https://platform.openai.com/docs/guides/text-generation/json-mode
"""

import os
import json
import random

from dotenv import load_dotenv
from  openai import OpenAI
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from messages import (
    CYCLE_CHECKER_MSG_SYS_CONTENT,
    EXPAN_MSG_SYS_CONTENT,
    INIT_MSG_SYS_CONTENT
)

load_dotenv()

DATA_SET = pd.read_csv('Dataset.csv')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def llm_model(sys_content: str, data: dict) -> json:
    """
    Sends a prompt to the OpenAI API and returns the llm response.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)

    user_content = f"Data for processing: {data} and return in json format."

    response = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        response_format={'type': 'json_object'},
        messages=[
            {
                'role': 'system',
                'content': sys_content
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]
    )

    llm_response = json.loads(response.choices[0].message.content)

    return llm_response


def initial_variable_selector(data_set) -> list[dict]:
    """
    Selects a random row from the dataset and feeds it to the LLM model.
    """

    # Get rand row (1 in 1552210).
    get_random_row = data_set.sample(n=1, random_state=1552210)

    # Removing columns with the Nan object.
    clean_row = get_random_row.dropna(axis=1)

    row_dict = clean_row.to_dict(orient='records')[0]

    llm_response = llm_model(INIT_MSG_SYS_CONTENT, {'data': row_dict})

    print(f"Initial variable selector:\n{llm_response}\n")

    return llm_response['data']


def expansion_generator(current_nod: str, other_nods: list) -> list[str]:
    """
    Using llm, generates cause and effect relationships.
    """

    llm_response = llm_model(
        EXPAN_MSG_SYS_CONTENT,
        {
            'current_nod': current_nod,
            'other_nods': other_nods
        }
    )

    print(f"\n{llm_response}\n")

    return llm_response['relationships']


def cycle_checker(graph: dict, visited_nod: str, nod: str) -> bool:
    """
    Using llm, checks for causal cycles that violate the DAG
    (directed acyclic graph) condition.
    """

    llm_response = llm_model(
        CYCLE_CHECKER_MSG_SYS_CONTENT,
        {
            'graph': graph,
            'visited_nod': visited_nod,
            'nod': nod
        }
    )

    print(f"\n{llm_response}\n")

    return llm_response[nod]


def generate_graph(data_set,
                   initial_variable_selector,
                   expansion_generator,
                   cycle_checker) -> dict:
    """
    This function generates a graph using a LLM and a set of rules.

    Returns:
    * A dictionary representing the generated graph.
    * Keys are variable names, values are lists of connected variables.
    """

    # Initialize empty graph.
    graph = {}

    # Receive cleaned and described data.
    frontier = initial_variable_selector(data_set)

    # It stores processed nodes.
    visited = []

    # Loop until frontier is empty.
    while frontier:
        visited_nod = frontier.pop(0)
        visited.append(visited_nod)

        # Generate potential expansions using the LLM.
        expansions = expansion_generator(visited_nod, frontier)

        # Check for cycles and add valid expansions to the graph or frontier.
        for nod in expansions:

            if cycle_checker(graph, visited_nod, nod) is False:
                graph.setdefault(visited_nod['name'], [])
                graph[visited_nod['name']].append(nod)

                # if nod not in frontier and nod not in visited:
                #     frontier.append(nod)

    return graph


def draw_a_graph(graph: dict) -> None:
    # Init graph
    G = nx.Graph()

    # Add nodes and edges
    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge)

    # Set graph parameters
    plt.figure(figsize=(8, 8))

    # Determine the location of nodes.
    pos = nx.spring_layout(G)

    # Graph visualization
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        arrows=True,
        arrowstyle='->',
        arrowsize=20
    )

    # Display graph
    plt.title("Causal Graph")
    plt.savefig('causal_graph')  # Save the plot to a file
    # plt.show()


def main() -> None:
    graph = generate_graph(
        DATA_SET,
        initial_variable_selector,
        expansion_generator,
        cycle_checker
    )

    print(f"Graph:\n{graph}\n")

    draw_a_graph(graph)


if __name__=='__main__':
    main()
