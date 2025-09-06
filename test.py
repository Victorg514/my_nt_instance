import transformers, inspect
from transformers import TrainingArguments

print("Transformers version :", transformers.__version__)
print("Class path           :", TrainingArguments.__module__)
print("File on disk         :", TrainingArguments.__module__.replace('.', '/') + '.py')
print("Init signature       :", inspect.signature(TrainingArguments.__init__))
