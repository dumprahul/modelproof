import ezkl
ezkl.gen_settings("simple_model.onnx")
ezkl.compile_circuit("simple_model.onnx", "simple_model.ezkl", "settings.json")
ezkl.setup("simple_model.ezkl", "vk.key", "pk.key", "kzg.srs")


