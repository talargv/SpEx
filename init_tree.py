from cart_based import CARTBased
from clique_based import CliqueBased
from graph_based import GraphBased
from centers_based import CentersBased

def init_tree_by_name(tree_name):
    if tree_name == 'cart':
        return CARTBased()
    elif tree_name == 'clique':
        return CliqueBased()
    elif tree_name == 'graph':
        return GraphBased()
    elif tree_name == 'emn':
        return CentersBased(algorithm='emn')
    elif tree_name == 'imm':
        return CentersBased(algorithm='imm')
    else:
        raise ValueError(f"Invalid tree name: {tree_name}")
    