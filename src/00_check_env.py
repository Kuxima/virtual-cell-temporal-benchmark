# -*- coding: utf-8 -*-
"""环境自检：GPU / 关键依赖 / BioMNI 工具层 import。
用法：在待测环境中 `python src/00_check_env.py [--biomni]`
"""
import sys


def main(check_biomni: bool = False):
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    import torch

    ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  CUDA available: {ok}")
    if ok:
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  capability: sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
        x = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
        print(f"  GPU matmul OK: {x.sum().item():.1f}")

    import scanpy as sc
    import anndata as ad

    print(f"scanpy {sc.__version__}, anndata {ad.__version__}")

    if check_biomni:
        import biomni
        from biomni.tool.genomics import (  # noqa: F401
            create_scvi_embeddings_scRNA,
            create_harmony_embeddings_scRNA,
            generate_embeddings_with_state,
            generate_transcriptformer_embeddings,
            get_uce_embeddings_scRNA,
            interspecies_gene_conversion,
        )
        print(f"biomni {biomni.__version__}: tool layer import OK")
        import scvi

        print(f"scvi-tools {scvi.__version__}")

    print("ENV CHECK PASSED")


if __name__ == "__main__":
    main(check_biomni="--biomni" in sys.argv)
