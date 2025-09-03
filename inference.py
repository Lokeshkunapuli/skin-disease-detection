"""
@author: denil gabani

"""
import os
from openvino.runtime import Core
import xml.etree.ElementTree as ET

class Network:

    def __init__(self):
        self.core = Core()
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.result = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        # Prefer explicit IR loading and validate IR version for compatibility
        xml_path = model
        bin_path = os.path.splitext(model)[0] + ".bin"

        # Validate that IR version is supported (OpenVINO 2025 requires IR v10+)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ir_version_str = root.attrib.get("version", "0")
            ir_version = int(ir_version_str) if ir_version_str.isdigit() else 0
        except ET.ParseError:
            # If parsing fails, proceed to let OpenVINO throw a detailed error
            ir_version = 0

        if ir_version and ir_version < 10:
            raise RuntimeError(
                f"Unsupported IR version {ir_version} in {os.path.basename(xml_path)}. "
                "This model was generated with a legacy OpenVINO (IR v10+ required). "
                "Please re-convert the original model to a modern IR or install an older OpenVINO runtime."
            )

        ov_model = self.core.read_model(model=xml_path, weights=bin_path)
        self.compiled_model = self.core.compile_model(model=ov_model, device_name=device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def get_input_shape(self):
        # Gets the input shape of the network
        return self.input_layer.shape

    def async_inference(self, image):
        # Use synchronous infer call with dict input
        self.result = self.compiled_model({self.input_layer: image})

    def wait(self):
        # Always return 0 (success) for synchronous inference
        return 0
    def extract_output(self):
        # Returns the results for the output layer of the network
        return self.result[self.output_layer]
# ...existing code...
