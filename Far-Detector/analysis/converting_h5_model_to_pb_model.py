#!/usr/bin/env python
# coding: utf-8
# # converting_h5_model_to_pb_model.py
# A. Wasit Yahaya 
# April 2025


#  $ PY37 converting_h5_model_to_pb_model.py <path_to_h5_file>



import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import sys

def convert_h5_to_pb(h5_model_path):
    # Validate input path
    if not os.path.exists(h5_model_path) or not h5_model_path.endswith('.h5'):
        raise ValueError("Please provide a valid .h5 model file path.")

    # Load the Keras model
    model = load_model(h5_model_path)

    # Prepare input signatures based on model's input shapes
    if isinstance(model.input_shape, list):
        input_signatures = [
            tf.TensorSpec(shape=[None, *shape[1:]], dtype=tf.float32)
            for shape in model.input_shape
        ]
    else:
        input_signatures = [tf.TensorSpec(shape=[None, *model.input_shape[1:]], dtype=tf.float32)]

    @tf.function(input_signature=input_signatures)
    def model_fn(*inputs):
        return model(inputs)

    # Get concrete function and freeze the model
    concrete_func = model_fn.get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(concrete_func)

    # Save the model to .pb format
    output_dir = os.path.dirname(h5_model_path)
    base_name = os.path.splitext(os.path.basename(h5_model_path))[0]
    pb_model_path = os.path.join(output_dir, f"{base_name}.pb")

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=output_dir,
                      name=f"{base_name}.pb",
                      as_text=False)

    print(f"Model successfully converted and saved as: {pb_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_h5_to_pb.py path/to/your_model.h5")
        sys.exit(1)

    h5_model_path = sys.argv[1]
    convert_h5_to_pb(h5_model_path)

