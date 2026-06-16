# Notes on Transformers, LoRA, and RAG

Transformers are neural network architectures that use self-attention to model relationships between tokens in a sequence. They are widely used in large language models because attention helps the model combine information from different parts of the input.

LoRA reduces the number of trainable parameters by injecting low-rank matrices into model weights. Instead of updating every parameter in a large model, LoRA trains small adapter matrices, which makes fine-tuning cheaper and easier to store.

RAG combines retrieval with generation to ground answers in external documents. In a RAG question answering system, the retriever finds relevant context first, and the language model generates an answer conditioned on that retrieved context.

RAG does not automatically guarantee factual correctness. A generated answer can still include unsupported claims if the retrieved context is incomplete, irrelevant, or ignored by the language model.
