use crate::ssd::offset_index::OffsetIndex;
use crate::ssd::weight_descriptor::{LayerSpec, LayerStructure, WeightRef};
use std::collections::BTreeMap;

pub struct SdxlUnetStructure {
    layer_specs: Vec<LayerSpec>,
}

impl SdxlUnetStructure {
    pub fn from_offset_index(index: &OffsetIndex) -> anyhow::Result<Self> {
        let mut groups: BTreeMap<String, Vec<WeightRef>> = BTreeMap::new();

        for name in index.tensor_names() {
            if !is_unet_tensor(name) {
                continue;
            }
            let prefix = layer_prefix(name);
            groups
                .entry(prefix)
                .or_default()
                .push(WeightRef::new(name.clone(), ""));
        }

        let layer_specs: Vec<LayerSpec> = {
            let mut entries: Vec<(String, Vec<WeightRef>)> = groups.into_iter().collect();
            entries.sort_by_key(|(name, _)| sort_key(name));
            entries
                .into_iter()
                .enumerate()
                .map(|(id, (name, weights))| LayerSpec::new(id, name, weights))
                .collect()
        };

        Ok(Self { layer_specs })
    }
}

impl LayerStructure for SdxlUnetStructure {
    fn layer_specs(&self) -> Vec<LayerSpec> {
        self.layer_specs.clone()
    }
}

fn is_unet_tensor(name: &str) -> bool {
    let prefixes = [
        "conv_in.",
        "down_blocks.",
        "mid_block.",
        "up_blocks.",
        "conv_norm_out.",
        "conv_out.",
        "time_embedding.",
    ];
    prefixes.iter().any(|p| name.starts_with(p))
}

fn layer_prefix(name: &str) -> String {
    if name.starts_with("conv_in.") {
        return "conv_in".into();
    }

    if name.starts_with("time_embedding.") {
        return "time_embedding".into();
    }

    if let Some(rest) = name.strip_prefix("down_blocks.") {
        let (d, rest) = split_dot(rest);
        if let Some(rest) = rest.strip_prefix("resnets.") {
            let (r, _) = split_dot(rest);
            return format!("down_blocks.{d}.resnets.{r}");
        }
        if let Some(rest) = rest.strip_prefix("attentions.") {
            let (a, rest) = split_dot(rest);
            if let Some(rest) = rest.strip_prefix("transformer_blocks.") {
                let (t, _) = split_dot(rest);
                if let Some(pos) = name.find(".attn1.")
                    .or_else(|| name.find(".attn2."))
                    .or_else(|| name.find(".ff."))
                {
                    return name[..pos].to_string();
                }
                return format!("down_blocks.{d}.attentions.{a}.transformer_blocks.{t}");
            }
            return format!("down_blocks.{d}.attentions.{a}");
        }
        if rest.starts_with("downsamplers.") {
            return format!("down_blocks.{d}.downsamplers");
        }
        return format!("down_blocks.{d}");
    }

    if let Some(rest) = name.strip_prefix("mid_block.") {
        if let Some(rest) = rest.strip_prefix("resnets.") {
            let (r, _) = split_dot(rest);
            return format!("mid_block.resnets.{r}");
        }
        if let Some(rest) = rest.strip_prefix("attentions.") {
            let (a, rest) = split_dot(rest);
            if let Some(rest) = rest.strip_prefix("transformer_blocks.") {
                let (t, _) = split_dot(rest);
                if let Some(pos) = name.find(".attn1.")
                    .or_else(|| name.find(".attn2."))
                    .or_else(|| name.find(".ff."))
                {
                    return name[..pos].to_string();
                }
                return format!("mid_block.attentions.{a}.transformer_blocks.{t}");
            }
            return format!("mid_block.attentions.{a}");
        }
    }

    if let Some(rest) = name.strip_prefix("up_blocks.") {
        let (u, rest) = split_dot(rest);
        if let Some(rest) = rest.strip_prefix("resnets.") {
            let (r, _) = split_dot(rest);
            return format!("up_blocks.{u}.resnets.{r}");
        }
        if let Some(rest) = rest.strip_prefix("attentions.") {
            let (a, rest) = split_dot(rest);
            if let Some(rest) = rest.strip_prefix("transformer_blocks.") {
                let (t, _) = split_dot(rest);
                if let Some(pos) = name.find(".attn1.")
                    .or_else(|| name.find(".attn2."))
                    .or_else(|| name.find(".ff."))
                {
                    return name[..pos].to_string();
                }
                return format!("up_blocks.{u}.attentions.{a}.transformer_blocks.{t}");
            }
            return format!("up_blocks.{u}.attentions.{a}");
        }
        if rest.starts_with("upsamplers.") {
            return format!("up_blocks.{u}.upsamplers");
        }
        return format!("up_blocks.{u}");
    }

    if name.starts_with("conv_norm_out.") {
        return "conv_norm_out".into();
    }

    if name.starts_with("conv_out.") {
        return "conv_out".into();
    }

    name.to_string()
}

fn sort_key(name: &str) -> i32 {
    if name == "time_embedding" {
        return -2;
    }
    if name == "conv_in" {
        return 0;
    }
    if name == "conv_norm_out" {
        return 30000;
    }
    if name == "conv_out" {
        return 30001;
    }

    if let Some(rest) = name.strip_prefix("down_blocks.") {
        return block_sort_key(rest, 100, false);
    }
    if let Some(rest) = name.strip_prefix("mid_block.") {
        return block_sort_key(rest, 10000, false);
    }
    if let Some(rest) = name.strip_prefix("up_blocks.") {
        return block_sort_key(rest, 20000, true);
    }

    99999
}

fn block_sort_key(rest: &str, block_base: i32, reverse: bool) -> i32 {
    let (block_str, rest) = split_dot(rest);
    let block: i32 = block_str.parse().unwrap_or(0);

    // ResNet blocks come first: block*1000 + resnet*10
    if let Some(rest) = rest.strip_prefix("resnets.") {
        let (r, _) = split_dot(rest);
        let r: i32 = r.parse().unwrap_or(0);
        let key = if reverse {
            block_base - block * 1000 + r * 10
        } else {
            block_base + block * 1000 + r * 10
        };
        return key;
    }

    // Attention parent (norm + proj_in + proj_out): block*1000 + 990 + attn*10 + 5
    if let Some(rest) = rest.strip_prefix("attentions.") {
        let (attn_str, rest) = split_dot(rest);
        let attn: i32 = attn_str.parse().unwrap_or(0);

        if let Some(rest) = rest.strip_prefix("transformer_blocks.") {
            let (tb, _) = split_dot(rest);
            let tb: i32 = tb.parse().unwrap_or(0);
            let key = if reverse {
                block_base - block * 1000 + 995 + attn * 10 + tb
            } else {
                block_base + block * 1000 + 995 + attn * 10 + tb
            };
            return key;
        }

        // Parent spatial group
        let key = if reverse {
            block_base - block * 1000 + 990 + attn * 10
        } else {
            block_base + block * 1000 + 990 + attn * 10
        };
        return key;
    }

    // Downsampler/upsampler comes last: block*1000 + 1999
    let key = if reverse {
        block_base - block * 1000 + 1999
    } else {
        block_base + block * 1000 + 1999
    };
    key
}

fn split_dot(s: &str) -> (&str, &str) {
    let dot = s.find('.').unwrap_or(s.len());
    (&s[..dot], if dot < s.len() { &s[dot + 1..] } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_prefixes() {
        assert_eq!(layer_prefix("conv_in.weight"), "conv_in");
        assert_eq!(layer_prefix("conv_in.bias"), "conv_in");

        assert_eq!(
            layer_prefix("down_blocks.0.resnets.0.norm1.weight"),
            "down_blocks.0.resnets.0"
        );
        assert_eq!(
            layer_prefix("down_blocks.0.resnets.1.conv1.weight"),
            "down_blocks.0.resnets.1"
        );

        let attn = layer_prefix("down_blocks.1.attentions.0.transformer_blocks.2.attn1.to_q.weight");
        assert!(attn.contains("down_blocks.1.attentions.0.transformer_blocks.2"));

        let ff = layer_prefix("down_blocks.2.attentions.0.transformer_blocks.5.ff.net.0.proj.weight");
        assert!(ff.contains("down_blocks.2.attentions.0.transformer_blocks.5"));

        assert_eq!(
            layer_prefix("up_blocks.0.resnets.1.norm1.weight"),
            "up_blocks.0.resnets.1"
        );

        let up_attn = layer_prefix(
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        );
        assert!(up_attn.contains("up_blocks.1.attentions.0.transformer_blocks.0"));

        let mid = layer_prefix(
            "mid_block.attentions.0.transformer_blocks.9.attn1.to_q.weight",
        );
        assert!(mid.contains("mid_block.attentions.0.transformer_blocks.9"));

        assert_eq!(layer_prefix("conv_norm_out.weight"), "conv_norm_out");
        assert_eq!(layer_prefix("conv_out.weight"), "conv_out");
        assert_eq!(
            layer_prefix("down_blocks.0.downsamplers.0.conv.weight"),
            "down_blocks.0.downsamplers"
        );
    }
}

