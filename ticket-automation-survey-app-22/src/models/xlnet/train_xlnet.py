from src.models.xlnet.xlnet import XLNetForClassification
from src.models.hierarchical_labeling.training_bert4c import run
from pathlib import Path

"""
- **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:
    - `"last"` -- Take the last token hidden state (like XLNet)
    - `"first"` -- Take the first token hidden state (like Bert)
    - `"mean"` -- Take the mean of all tokens hidden states
    - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
    - `"attn"` -- Not implemented now, use multi-head attention
- **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
- **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
  (otherwise to `config.hidden_size`).
- **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
  another string or `None` will add no activation.
- **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
- **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
"""
if __name__ == "__main__":

    # tests = [f"xlnet_last.yml", f"xlnet_first.yml", f"xlnet_mean.yml", f"xlnet_cls_index.yml"]
    tests = [f"xlnet_last_financial.yml"]

    mod_cls = XLNetForClassification

    for name in tests:
        config_path = Path("src") / "models" / "xlnet" / "configs"
        run(name, mod_cls, full_path=config_path)
