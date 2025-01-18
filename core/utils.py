import graphviz

def get_edges_nodes(atom):
    nodes, edges = set(), set()

    def build_graph(atom):
        if atom not in nodes:
            nodes.add(atom)
            for child in atom._prev:
                edges.add((child, atom))
                build_graph(child)

    build_graph(atom)
    return nodes, edges


def visualize_graph(atom):
    dot = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = get_edges_nodes(atom)

    # Add nodes and edges to the graph
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph create a rectangle node with lable and value
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # create operation node
            
            dot.node(name=uid+n._op, label=n._op)
            # create edge between operation node and value node
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))+ n2._op)

    return dot