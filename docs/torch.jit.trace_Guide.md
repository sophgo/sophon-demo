[简体中文](./torch.jit.trace_Guide.md) | [English](./torch.jit.trace_Guide_EN.md)

 ## 1. 什么是JIT（torch.jit）？

答：JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。

 ## 2. 如何得到JIT模型？

答：在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace得到；traced_model=torch.jit.trace(python_model, torch.rand(input_shape))，然后再用traced_model.save(‘jit.pt’)保存下来。

## 3. 为什么不能使用torch.jit.script得到JIT模型？

答：BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，但torch.jit.script可以产生这类模型，而torch.jit.trace却不可以，仅跟踪和记录张量上的操作，不会记录任何控制流操作。

## 4. 为什么不能是GPU模型？

答：BMNETP的编译过程不支持。

## 5. 如何将GPU模型转成CPU模型？

答：在加载PyTorch的Python模型时，使用map_location参数：torch.load(python_model, map_location = ‘cpu’)。