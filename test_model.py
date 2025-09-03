from openvino.runtime import Core
import os

core = Core()
xml_path = "model/model_tf.xml"
bin_path = os.path.splitext(xml_path)[0] + ".bin"
try:
    model = core.read_model(model=xml_path, weights=bin_path)
    compiled = core.compile_model(model, device_name="CPU")
    input_shape = compiled.input(0).shape
    output_shape = compiled.output(0).shape
    print("Model loaded successfully! Input:", input_shape, "Output:", output_shape)
except Exception as e:
    print("Error loading model:", e)