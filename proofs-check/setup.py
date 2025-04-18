import ezkl

# First generate the settings
ezkl.gen_settings("hot.onnx", "settings.json")

# Then calibrate with the generated settings
ezkl.calibrate_settings("hot.onnx", "settings.json", target="accuracy")

# Finally compile the circuit
ezkl.compile_circuit("hot.onnx", "hot.ezkl", "settings.json")


