[简体中文](./torch.jit.trace_Guide.md) | [English](./torch.jit.trace_Guide_EN.md)

## 1.What is JIT (torch.jit)?

Answer: JIT (Just-In-Time) is a set of compilation tools used to bridge the gap between PyTorch research and production. It allows the creation of models that can run without relying on the Python interpreter and can be more aggressively optimized.

## 2.How to get a JIT model?

Answer: In the case of an existing PyTorch Python model (with the base class torch.nn.Module), a JIT model can be obtained using torch.jit.trace: traced_model = torch.jit.trace(python_model, torch.rand(input_shape)), and then save it using traced_model.save('jit.pt'). Attention When loading a PyTorch model before trace, use the map_location parameter: torch.load(python_model, map_location='cpu').

## 3.Why can't we use torch.jit.script to get a JIT model?

Answer: BMNETP does not currently support JIT models with control flow operations (such as if statements or loops), but torch.jit.script can generate such models, while torch.jit.trace can only trace and record operations on tensors and does not record any control flow operations.
