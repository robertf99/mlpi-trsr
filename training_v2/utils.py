from IPython.display import clear_output, Image, display, HTML
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.compat.v1.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == "Const":
            tensor = n.attr["value"].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>" % size, "utf-8")
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, "graph_def"):
        graph_def = graph_def.graph_def
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(
        data=repr(str(strip_def)), id="graph" + str(np.random.rand())
    )

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(
        code.replace('"', "&quot;")
    )
    display(HTML(iframe))


def save_strip_pbtxt(export_dir: str):
    """
    extract graphDef from exported model.pb, strip it and save as seperate model.pbtxt for visualisation
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        graph = tf.compat.v1.saved_model.loader.load(sess, ["serve"], export_dir)
    strip_def = strip_consts(graph.graph_def)
    tf.io.write_graph(strip_def, export_dir, "model.pbtxt", as_text=True)


def add_custom_summary(tag, value):
    s = tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
    )
    return s
