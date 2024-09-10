"""
Module for describing text queries for LLM.
"""

INIT_MSG_SYS_CONTENT = """
* Analyze all elements in {data}.
* Step through the objects in {data} and return them in a more detailed format.
* Each json object must contain the following keys:
    1) name: The key representing the name or identifier of the element.
    2) value: The value of the element.
    3) description: A textual description of the element, explaining its value and role.
"""

EXPAN_MSG_SYS_CONTENT = """
* Find the causal-relationship between the nodes from {current_node}
  to each node in the list {other_nodes} and return a list of nods
  with which {current_node} has a causal relationship.
* Return an object where:
    1) name: {current_node} name.
    2) relationships: List of nod names with which {current_node} has a causal relationship.
"""

CYCLE_CHECKER_MSG_SYS_CONTENT = """
* Analyze {nod}, {graph}, and {visited_nod}, where:
    * {visited_nod} has a causal relationship with {nod}.
    * {graph} represents the already checked and ordered connections.

* Check for the presence of causal loops that violate the DAG (Directed Acyclic Graph) condition.
* Return an object where:
    * key = The name of {nod}.
    * value = A boolean value.
"""
