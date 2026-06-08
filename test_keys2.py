import sys
sys.path.append('src')

from maxtext.checkpoint_conversion.to_maxtext import main
try:
    main(["--model_name=gemma3-4b", "--eager_load_method=transformers"], lazy_load_tensors=False)
except Exception as e:
    print(e)
