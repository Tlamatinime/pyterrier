from . import Transformer, ops, rewrite
from typing import List

def _check_graphviz(name="this functionality"):
    try:
        import graphviz
    except ImportError:
        raise ImportError("You must pip install graphviz to use %s" % name)
    
def render(pipe : Transformer):
    """
    Returns a `graphviz <https://graphviz.readthedocs.io/en/stable/>`_ Graph visualisation of 
    the provided transformer pipeline, which can be easily displayed in a Jupyter/Colab Notebook.

    Example Usage in a Notebook::
        
        pipe = (dph % 5) >> qe >> dph
        pt.debug.render(pipe)

    """


    _COMPOSE_AS_NODE = False

    _check_graphviz("pt.debug.render()")

    if not isinstance(pipe, Transformer):
        raise TypeError("%s is not a pt.Transformer" % type(pipe))

    def traverse(pipe, dot, depth=0):
        if isinstance(pipe, ops.ScalarProductTransformer):
            id_node = "*"
            dot.node(id_node, label="*", shape='diamond')
            model0 = traverse(pipe.transformer, dot, depth=depth+1)
            dot.edge(model0, id_node)
            dot.node(id_node + "_cutoff", label=str(pipe.scalar), shape='rect')
            dot.edge(id_node + "_cutoff", id_node)
            return id_node
        elif isinstance(pipe, ops.RankCutoffTransformer):
            id_node = "%"
            dot.node(id_node, label="%", shape='diamond')
            model0 = traverse(pipe.transformer, dot, depth=depth+1)
            dot.edge(model0, id_node)
            dot.node(id_node + "_cutoff", label=str(pipe.cutoff), shape='rect')
            dot.edge(id_node + "_cutoff", id_node)
            return id_node
        elif isinstance(pipe, ops.FeatureUnionPipeline):
            assert len(pipe.models) == 2

            model0 = traverse(pipe.models[0], dot, depth=depth+1)
            model1 = traverse(pipe.models[1], dot, depth=depth+1)

            id_compose = "**" + str(depth)
            #print("drawing ** %s joining %s with %s"   % (id_compose, model0, model1))

            #dot.edge(model0, model1, label="**")
            dot.node(id_compose, label="**", shape='diamond')
            dot.edge(id_compose, model0)
            dot.edge(id_compose, model1)
            return id_compose

        elif isinstance(pipe, ops.ComposedPipeline):
            assert len(pipe.models) == 2
            

            model0 = traverse(pipe.models[0], dot, depth=depth+1)
            model1 = traverse(pipe.models[1], dot, depth=depth+1)

            id_compose = ">>" + str(depth)
            #print("drawing >> %s joining %s with %s"   % (id_compose, model0, model1))

            if _COMPOSE_AS_NODE:
                dot.node(id_compose, label=">>", shape='cds')
                dot.edge(id_compose, model0)
                dot.edge(id_compose, model1)
                return id_compose
        
            else:
                dot.edge(model0, model1, label=">>")
                return model1
        
        else:

            id_node = str(depth) + str(pipe)
            label=str(pipe)
            if isinstance(pipe, rewrite.QueryExpansion):
                label = 'QE' 
            dot.node(id_node, label=label)
            #print("drawing node %s" % id_node)
            return id_node

    from graphviz import Digraph
    dot = Digraph()
    traverse(pipe, dot)
    return dot


def print_columns(by_query : bool = False, message : str = None) -> Transformer:
    """
    Returns a transformer that can be inserted into pipelines that can print the column names of the dataframe
    at this stage in the pipeline:

    Arguments:
     - by_query(bool): whether to display for each query. Defaults to False.
     - message(str): whether to display a message before printing. Defaults to None, which means no message. This
       is useful when ``print_columns()`` is being used multiple times within a pipeline 
     

    Example::
    
        pipe = (
            bm25
            >> pt.debug.print_columns() 
            >> pt.rewrite.RM3() 
            >> pt.debug.print_columns()
            bm25

    When the above pipeline is executed, two sets of columns will be displayed
     - `["qid", "query", "docno", "rank", "score"]`  - the output of BM25, a ranking of documents
     - `["qid", "query", "query_0"]`   - the output of RM3, a reformulated query
    
        
    """
    import pyterrier as pt
    def _do_print(df):
        if message is not None:
            print(message)
        print(df.columns)
        return df
    return pt.apply.by_query(_do_print) if by_query else pt.apply.generic(_do_print) 

def print_num_rows(
        by_query = True, 
        msg="num_rows") -> Transformer:
    """
    Returns a transformer that can be inserted into pipelines that can print the number of rows names of the dataframe
    at this stage in the pipeline:

    Arguments:
     - by_query(bool): whether to display for each query. Defaults to True.
     - message(str): whether to display a message before printing. Defaults to "num_rows". This
       is useful when ``print_columns()`` is being used multiple times within a pipeline 
     
    Example::
    
        pipe = (
            bm25
            >> pt.debug.print_num_rows() 
            >> pt.rewrite.RM3() 
            >> pt.debug.print_num_rows()
            bm25

    When the above pipeline is executed, the following output will be displayed
     - `num_rows 1: 1000` - the output of BM25, a ranking of documents
     - `num_rows 1: 1` - the output of RM3, the reformulated query
    
        
    """

    import pyterrier as pt
    def _print_qid(df):
        qid = df.iloc[0].qid
        print("%s %s: %d" % (msg, qid, len(df)))
        return df

    def _print(df):
        print("%s: %d" % (msg, len(df)))
        return df

    if by_query:
        return pt.apply.by_query(_print_qid, add_ranks=False)
    else:
        return pt.apply.generic(_print, add_ranks=False)

def print_rows(
        by_query : bool = True, 
        jupyter: bool = True, 
        head : int = 2, 
        message : str = None, 
        columns : List[str] = None) -> Transformer:
    """
    Returns a transformer that can be inserted into pipelines that can print some of the dataframe
    at this stage in the pipeline:

    Arguments:
     - by_query(bool): whether to display for each query. Defaults to True.
     - jupyter(bool): Whether to use IPython's display function to display the dataframe. Defaults to True.
     - head(int): The number of rows to display. None means all rows.
     - columns(List[str]): Limit the columns for which data is displayed. Default of None displays all columns.
     - message(str): whether to display a message before printing. Defaults to None, which means no message. This
       is useful when ``print_rows()`` is being used multiple times within a pipeline 

    Example::

        pipe = (
            bm25
            >> pt.debug.print_rows() 
            >> pt.rewrite.RM3() 
            >> pt.debug.print_rows()
            bm25
     
    """
    import pyterrier as pt
    def _do_print(df):
        if message is not None:
            print(message)
        render = df if head is None else df.head(head)
        if columns is not None:
            render = render[columns]
        if jupyter:
            from IPython.display import display
            display(render)
        else:
            print(render)
        return df
    return pt.apply.by_query(_do_print) if by_query else pt.apply.generic(_do_print) 